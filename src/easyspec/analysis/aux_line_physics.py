"""
Secondary functions of line_physics(): physical quantities of the fitted lines (flux, equivalent width, FWHM, line dispersion and their
uncertainties), computed in two ways:

- From the line models: each quantity is computed for each sample of the MCMC posterior distribution of the line parameters (and of the
  redshift), and its value and asymmetric errors are the median and the 16% and 84% quantiles. This propagates the full posterior, including
  the correlations between the parameters (e.g. amplitude and width), without any linear approximation.
- From the line profile (model-independent): the quantities are measured directly on the continuum-subtracted data (minus the other
  components of a blend, if any), and their uncertainties are obtained with a Monte Carlo simulation of the noise of the data (Gaussian noise
  with the uncertainty of each pixel).
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import wofz
from scipy.ndimage import gaussian_filter1d

from .line_models import model_Gauss, model_Lorentz, model_Voigt, model_skewed_gaussian, model_skewed_lorentzian

SPEED_OF_LIGHT = 299792.458   # km/s
FWHM_TO_SIGMA = 1/(2*np.sqrt(2*np.log(2)))

MODEL_FUNCTIONS = {"Gaussian": model_Gauss, "Lorentz": model_Lorentz, "Voigt": model_Voigt, "Skewedgaussian": model_skewed_gaussian,
                   "Skewedlorentzian": model_skewed_lorentzian}


def summarize(samples, quantiles=(0.16, 0.5, 0.84)):

    """[median, error_down, error_up] of a set of samples (NaNs ignored; all NaN if no finite sample)."""

    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return [np.nan, np.nan, np.nan]
    low, median, high = np.quantile(samples, quantiles)
    return [float(median), float(median - low), float(high - median)]


def summarize_around(value, samples, quantiles=(0.16, 0.5, 0.84)):

    """[value, error_down, error_up], with the errors from the quantiles of the Monte Carlo samples relative to their median."""

    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0 or not np.isfinite(value):
        return [float(value), np.nan, np.nan]
    q_low, q_median, q_high = np.quantile(samples, quantiles)
    return [float(value), float(q_median - q_low), float(q_high - q_median)]


def split_normal_samples(values, errors, n_samples, rng):

    """Samples of independent parameters with asymmetric errors [[down, up], ...] (split normal distributions)."""

    values = np.asarray(values, dtype=float)
    errors = np.abs(np.asarray(errors, dtype=float))
    normal = rng.standard_normal((n_samples, len(values)))
    scale = np.where(normal < 0, errors[:, 0], errors[:, 1])
    return values + normal*scale


def voigt_fwhm(fwhm_lorentz, fwhm_gauss):
    """FWHM of a Voigt profile (Olivero & Longbothum 1977, accuracy ~0.02%)."""
    return 0.5346*fwhm_lorentz + np.sqrt(0.2166*fwhm_lorentz**2 + fwhm_gauss**2)


def error_propagation_voigt(fwhm_lorentz, fwhm_gauss, error_lorentz, error_gauss):
    """Linear propagation of independent errors to the Voigt FWHM (derivatives of voigt_fwhm)."""
    root = np.sqrt(0.2166*fwhm_lorentz**2 + fwhm_gauss**2)
    derivative_lorentz = 0.5346 + 0.2166*fwhm_lorentz/root
    derivative_gauss = fwhm_gauss/root
    return np.sqrt((derivative_lorentz*error_lorentz)**2 + (derivative_gauss*error_gauss)**2)


def _numerical_profile(model, theta, fwhm_guess):
    """Grid (dense near the line, sparse in the far wings, up to +-100 FWHM) and the model evaluated on it."""
    t = np.concatenate([np.linspace(-100, -6, 300, endpoint=False), np.linspace(-6, 6, 2401, endpoint=False), np.linspace(6, 100, 301)])
    x = theta[0] + t*fwhm_guess
    return x, model(theta, x)


def _half_maximum_width(x, y):
    """FWHM of a sampled profile: half-maximum crossings (linear interpolation) on both sides of the maximum of |y|."""
    absolute = np.abs(y)
    peak = int(np.argmax(absolute))
    half = absolute[peak]/2
    left = peak
    while left > 0 and absolute[left] > half:
        left -= 1
    right = peak
    while right < len(y) - 1 and absolute[right] > half:
        right += 1
    if absolute[left] > half or absolute[right] > half:
        return np.nan
    x_left = np.interp(half, [absolute[left], absolute[left + 1]], [x[left], x[left + 1]])
    x_right = np.interp(half, [absolute[right], absolute[right - 1]], [x[right], x[right - 1]])
    return float(x_right - x_left)


def _crossings(x, profile, top, half):
    """First crossings (x_left, x_right) of the level 'half' on both sides of the pixel 'top' (linear interpolation); NaNs if not found."""
    left = top
    while left > 0 and profile[left] > half:
        left -= 1
    right = top
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    if profile[left] > half or profile[right] > half:
        return np.nan, np.nan
    x_left = np.interp(half, [profile[left], profile[left + 1]], [x[left], x[left + 1]])
    x_right = np.interp(half, [profile[right], profile[right - 1]], [x[right], x[right - 1]])
    return float(x_left), float(x_right)


def _crossing_width(x, profile, top, half):
    """Distance between the first crossings of the level 'half' on both sides of the pixel 'top'."""
    x_left, x_right = _crossings(x, profile, top, half)
    return x_right - x_left


def profile_peak(x, profile):
    """
    Pixel and value of the maximum of a (positive) line profile: the pixel is the maximum of the profile smoothed over 3 pixels, and the
    value is the vertex of a parabola through the 5 pixels around it (the maximum of the noisy profile itself is biased upwards).
    """
    if len(profile) < 5:
        top = int(np.argmax(profile))
        return top, float(profile[top])
    smooth = np.convolve(profile, np.ones(3)/3, mode="same")
    top = int(np.clip(np.argmax(smooth[1:-1]) + 1, 2, len(profile) - 3))
    coefficients = np.polyfit(x[top - 2:top + 3] - x[top], profile[top - 2:top + 3], 2)
    vertex = -coefficients[1]/(2*coefficients[0]) if coefficients[0] < 0 else 0.0
    return top, float(np.polyval(coefficients, np.clip(vertex, x[top - 2] - x[top], x[top + 2] - x[top])))


def smoothing_sigma(x, y, error, target_snr=15.0, max_fraction=0.35):

    """
    Width (sigma, in pixels) of the Gaussian kernel used to smooth a noisy line profile before measuring its FWHM (0 = no smoothing).

    On a noisy profile, the first point below half the maximum is often a noise dip close to the peak, and the maximum itself is biased
    upwards, so the FWHM is underestimated. The profile is smoothed just enough to bring its peak signal-to-noise ratio to ~target_snr, but
    never with a kernel FWHM larger than max_fraction times the width of the line (the interquartile range of its cumulative flux, which is
    the FWHM for a Lorentzian and 0.57 FWHM for a Gaussian). The broadening by the kernel is then removed in quadrature (exact for Gaussian
    lines; for Lorentzian lines the result is at most ~3% too large).
    """

    if len(y) < 7:
        return 0.0
    sign = 1.0 if trapezoid(y, x) >= 0 else -1.0
    profile = sign*np.asarray(y, dtype=float)
    noise = float(np.median(np.abs(error)))
    peak = float(np.max(np.convolve(profile, np.ones(3)/3, mode="same")[1:-1]))
    if not (noise > 0 and peak > 0):
        return 0.0
    snr = peak/noise
    if snr >= target_snr:
        return 0.0
    sigma_needed = (target_snr/snr)**2/(2*np.sqrt(np.pi))     # white noise is reduced by sqrt(2 sqrt(pi) sigma) by a Gaussian kernel
    cumulative = np.concatenate([[0.0], np.cumsum(0.5*(np.clip(profile[1:], 0, None) + np.clip(profile[:-1], 0, None))*np.diff(x))])
    if cumulative[-1] <= 0:
        return 0.0
    q25, q75 = np.interp([0.25, 0.75], cumulative/cumulative[-1], x)
    width_pixels = (q75 - q25)/np.median(np.diff(x))
    sigma = min(sigma_needed, max_fraction*width_pixels*FWHM_TO_SIGMA)
    return float(sigma) if sigma >= 0.5 else 0.0


def fwhm_points(x, y, smoothing=0.0):
    """
    The points used to measure the FWHM of a continuum-subtracted profile: (x_left, x_right, half maximum, maximum), with the sign of the line
    (negative values for absorption lines). The FWHM is x_right - x_left (observed frame, before any correction). If smoothing > 0 (sigma of
    a Gaussian kernel, in pixels), the points are measured on the smoothed profile, whose maximum is used directly.
    """
    sign = 1.0 if trapezoid(y, x) >= 0 else -1.0
    if smoothing > 0:
        smooth = gaussian_filter1d(sign*np.asarray(y, dtype=float), smoothing, mode="nearest")
        top = int(np.argmax(smooth))
        peak_value = float(smooth[top])
        x_left, x_right = _crossings(x, smooth, top, peak_value/2)
        return x_left, x_right, sign*peak_value/2, sign*peak_value
    top, peak_value = profile_peak(x, sign*y)
    x_left, x_right = _crossings(x, sign*y, top, peak_value/2)
    return x_left, x_right, sign*peak_value/2, sign*peak_value


def kernel_corrected_fwhm(fwhm, smoothing, pixel):
    """FWHM measured on a profile smoothed with a Gaussian kernel of sigma = smoothing pixels, with the kernel broadening removed in quadrature."""
    if not smoothing:
        return fwhm
    kernel_fwhm = smoothing*pixel/FWHM_TO_SIGMA
    return np.sqrt(np.clip(np.asarray(fwhm)**2 - kernel_fwhm**2, 0, None))


def truncated_dispersion(model_name, theta, limits, n_points=801):
    """Line dispersion (square root of the second moment) of a line model between the wavelengths limits = (lower, upper)."""
    grid = np.linspace(limits[0], limits[1], n_points)
    y = MODEL_FUNCTIONS[model_name](theta, grid)
    flux = trapezoid(y, grid)
    if flux == 0:
        return np.nan
    centroid = trapezoid(grid*y, grid)/flux
    second_moment = trapezoid((grid - centroid)**2*y, grid)/flux
    return float(np.sqrt(second_moment)) if second_moment > 0 else np.nan


def lorentzian_fwhm_from_voigt(fwhm_voigt, fwhm_gauss):
    """
    Lorentzian FWHM of a Voigt profile with total FWHM fwhm_voigt and Gaussian FWHM fwhm_gauss (inverse of voigt_fwhm). It removes the
    instrumental broadening (Gaussian) from the FWHM of a line with Lorentzian wings; 0 if fwhm_voigt <= fwhm_gauss.
    """
    a, b = 0.5346, 0.2166
    fwhm_voigt, fwhm_gauss = np.asarray(fwhm_voigt, dtype=float), np.asarray(fwhm_gauss, dtype=float)
    discriminant = np.clip((a*fwhm_voigt)**2 - (a**2 - b)*(fwhm_voigt**2 - fwhm_gauss**2), 0, None)
    lorentz = (a*fwhm_voigt - np.sqrt(discriminant))/(a**2 - b)
    return np.where(fwhm_voigt > fwhm_gauss, lorentz, 0.0)


def remove_instrumental_broadening(model_name, fwhm, dispersion, instrumental_fwhm, parameter_samples=None):
    """
    Intrinsic FWHM and line dispersion of line models, given the instrumental resolution (a Gaussian of FWHM instrumental_fwhm):

    - Gaussian (and skewed Gaussian): FWHM subtracted in quadrature (exact for Gaussians);
    - Lorentzian (and skewed Lorentzian): the fitted profile is approximately the Voigt profile of the intrinsic Lorentzian and the instrumental
      Gaussian, so the Lorentzian FWHM is obtained by inverting the Voigt FWHM formula (subtracting in quadrature would overcorrect);
    - Voigt: the instrumental FWHM is removed in quadrature from the Gaussian component only, and the Voigt FWHM is recomputed.

    The line dispersion is corrected in quadrature for all models (the variances of convolved profiles add up).
    """

    if not instrumental_fwhm:
        return fwhm, dispersion
    if model_name in ("Lorentz", "Skewedlorentzian"):
        fwhm = lorentzian_fwhm_from_voigt(fwhm, instrumental_fwhm)
    elif model_name == "Voigt" and parameter_samples is not None:
        fwhm_gauss = np.sqrt(np.clip(parameter_samples[:, 3]**2 - instrumental_fwhm**2, 0, None))
        fwhm = voigt_fwhm(parameter_samples[:, 2], fwhm_gauss)
    else:
        fwhm = np.sqrt(np.clip(np.asarray(fwhm)**2 - instrumental_fwhm**2, 0, None))
    dispersion = np.sqrt(np.clip(np.asarray(dispersion)**2 - (instrumental_fwhm*FWHM_TO_SIGMA)**2, 0, None))
    return fwhm, dispersion


def model_quantities(model_name, theta, dispersion_limits=None):

    """
    Flux (integral of the model), FWHM and line dispersion (square root of the second moment) of one line component, in the observed frame.

    The flux and FWHM are analytic for the Gaussian, Lorentzian and Voigt profiles. The second moment of profiles with Lorentzian wings
    (Lorentzian, Voigt and skewed Lorentzian) diverges: their line dispersion is computed between dispersion_limits (the integration limits of
    the profile measurement, which makes it comparable to the profile dispersion), or is np.nan if these are not given.
    """

    amplitude = theta[1]
    if model_name == "Gaussian":
        fwhm = theta[2]
        return amplitude*fwhm*np.sqrt(np.pi/(4*np.log(2))), fwhm, fwhm*FWHM_TO_SIGMA
    truncated = (lambda: truncated_dispersion(model_name, theta, dispersion_limits)) if dispersion_limits is not None else (lambda: np.nan)
    if model_name == "Lorentz":
        fwhm = theta[2]
        return amplitude*np.pi*fwhm/2, fwhm, truncated()
    if model_name == "Voigt":
        fwhm_lorentz, fwhm_gauss = theta[2], theta[3]
        sigma, gamma = fwhm_gauss*FWHM_TO_SIGMA, fwhm_lorentz/2
        peak_of_unit_area_profile = wofz(1j*gamma/(sigma*np.sqrt(2))).real/(sigma*np.sqrt(2*np.pi))
        return amplitude/peak_of_unit_area_profile, voigt_fwhm(fwhm_lorentz, fwhm_gauss), truncated()
    # Skewed models: numerical integration
    model = MODEL_FUNCTIONS[model_name]
    x, y = _numerical_profile(model, theta, theta[2])
    flux = trapezoid(y, x)
    fwhm = _half_maximum_width(x, y)
    if model_name == "Skewedlorentzian":
        return flux, fwhm, truncated()
    centroid = trapezoid(x*y, x)/flux
    return flux, fwhm, float(np.sqrt(max(trapezoid((x - centroid)**2*y, x)/flux, 0.0)))


def model_physics(model_name, parameter_samples, z_samples, continuum_function, instrumental_fwhm=None, max_samples=2000, rng=None,
                  dispersion_limits=None):

    """
    Physical quantities of one line component from its posterior samples.

    Parameters
    ----------
    model_name: string
    parameter_samples: array (n_samples, n_parameters)
        Posterior samples of [Location, Amplitude (physical units), widths...].
    z_samples: array (n_samples,)
        Redshift samples (used to convert to the rest frame).
    continuum_function: callable
        Continuum flux density at a given observed wavelength.
    instrumental_fwhm: float
        Optional. Instrumental FWHM (observed frame, Angstrom), removed from the FWHM (see remove_instrumental_broadening) and, in
        quadrature, from the line dispersion.
    dispersion_limits: (float, float)
        Optional. Observed wavelengths between which the line dispersion of models with Lorentzian wings is computed (see model_quantities).

    Returns
    -------
    dict with [value, error_down, error_up] of: flux, equivalent_width_rest, fwhm_rest, dispersion_rest, dispersion_velocity, fwhm_velocity
    """

    rng = np.random.default_rng() if rng is None else rng
    n = len(parameter_samples)
    index = rng.choice(n, size=min(n, max_samples), replace=False) if n > max_samples else np.arange(n)
    flux, fwhm, dispersion = np.array([model_quantities(model_name, theta, dispersion_limits) for theta in parameter_samples[index]]).T
    location = parameter_samples[index, 0]
    z = np.asarray(z_samples, dtype=float)[index] if np.ndim(z_samples) else np.full(len(index), float(z_samples))
    fwhm, dispersion = remove_instrumental_broadening(model_name, fwhm, dispersion, instrumental_fwhm, parameter_samples[index])
    continuum = continuum_function(location)
    equivalent_width = -flux/continuum   # positive for absorption lines (easyspec convention)
    return {"flux": summarize(flux),
            "equivalent_width_rest": summarize(equivalent_width/(1 + z)),
            "fwhm_rest": summarize(fwhm/(1 + z)),
            "dispersion_rest": summarize(dispersion/(1 + z)),
            "dispersion_velocity": summarize(SPEED_OF_LIGHT*dispersion/location),
            "fwhm_velocity": summarize(SPEED_OF_LIGHT*fwhm/location)}


def integration_limits(x, y, noise, peak_index, lower_bound, upper_bound):

    """
    Integration limits of a line profile: starting at the peak, the profile (smoothed over 3 pixels) is followed outwards until it reaches the
    noise level (1 sigma) or changes sign, and then extended by 50% to include the faint wings, within [lower_bound, upper_bound].
    """

    sign = np.sign(y[peak_index]) if y[peak_index] != 0 else 1.0
    smooth = np.convolve(sign*y, np.ones(3)/3, mode="same")
    level = np.asarray(noise)*np.ones(len(y))
    left = peak_index
    while left > 0 and x[left - 1] >= lower_bound and smooth[left - 1] > level[left - 1]:
        left -= 1
    right = peak_index
    while right < len(y) - 1 and x[right + 1] <= upper_bound and smooth[right + 1] > level[right + 1]:
        right += 1
    # The faint wings below the noise level are part of the line: the limits are extended by 50% of the half-extent found above (for a
    # Gaussian line with a peak S/N of ~50, this moves the limits from ~2.8 to ~4.2 sigma, which avoids underestimating the line dispersion).
    extension_left = int(np.ceil(0.5*(peak_index - left)))
    extension_right = int(np.ceil(0.5*(right - peak_index)))
    left = left - extension_left - 1
    right = right + extension_right + 1
    while left < 0 or x[max(left, 0)] < lower_bound:
        left += 1
    while right > len(y) - 1 or x[min(right, len(y) - 1)] > upper_bound:
        right -= 1
    return left, right


def profile_quantities(x, y, continuum, smoothing=0.0):

    """
    Model-independent quantities of a continuum-subtracted line profile y(x) between its first and last pixels: flux, equivalent width
    (integral of (Fc - F)/Fc, positive for absorption), centroid, line dispersion (Peterson et al. 2004, eqs. 4-5) and FWHM. If smoothing > 0,
    the FWHM is measured on the profile smoothed with a Gaussian kernel of sigma = smoothing pixels, and corrected for its broadening.
    """

    flux = trapezoid(y, x)
    equivalent_width = -trapezoid(y/continuum, x)
    if flux == 0:
        return flux, equivalent_width, np.nan, np.nan, np.nan
    centroid = trapezoid(x*y, x)/flux
    second_moment = trapezoid((x - centroid)**2*y, x)/flux
    dispersion = np.sqrt(second_moment) if second_moment > 0 else np.nan
    # FWHM: without smoothing (high S/N), the peak value is estimated with a parabola through the 5 pixels around the maximum, and the
    # half-maximum crossings are interpolated on the profile itself; noisy profiles are smoothed first (see smoothing_sigma).
    x_left, x_right, _, _ = fwhm_points(x, y, smoothing)
    fwhm = float(kernel_corrected_fwhm(x_right - x_left, smoothing, np.median(np.diff(x))))
    return flux, equivalent_width, centroid, dispersion, fwhm


def profile_physics(x, y, y_error, continuum, z_samples, n_realizations=500, instrumental_fwhm=None, rng=None):

    """
    Model-independent physical quantities of a line profile, with Monte Carlo uncertainties (n_realizations of Gaussian noise with
    the uncertainty y_error of each pixel). The redshift samples convert the quantities to the rest frame. For noisy profiles, the FWHM is
    measured after a light smoothing (see smoothing_sigma), the same in all realizations.

    Returns
    -------
    dict with [value, error_down, error_up] of: flux, equivalent_width_rest, fwhm_rest, dispersion_rest, dispersion_velocity, centroid
    """

    rng = np.random.default_rng() if rng is None else rng
    z = np.asarray(z_samples, dtype=float)
    z_draw = rng.choice(z, size=n_realizations) if np.ndim(z) and len(z) else np.full(n_realizations, float(z))
    z_value = float(np.median(z_draw))

    def corrected(fwhm, dispersion):
        if instrumental_fwhm:
            fwhm = np.sqrt(np.clip(fwhm**2 - instrumental_fwhm**2, 0, None))
            dispersion = np.sqrt(np.clip(dispersion**2 - (instrumental_fwhm*FWHM_TO_SIGMA)**2, 0, None))
        return fwhm, dispersion

    smoothing = smoothing_sigma(x, y, y_error)
    flux, ew, centroid, dispersion, fwhm = profile_quantities(x, y, continuum, smoothing)
    # For the plots (no instrumental correction; the FWHM points are those of the smoothed profile, if smoothed):
    measured = {"fwhm_points": fwhm_points(x, y, smoothing), "dispersion_observed": dispersion, "fwhm_observed": fwhm,
                "fwhm_smoothing_pixels": smoothing,
                "smoothed_profile": gaussian_filter1d(np.asarray(y, dtype=float), smoothing, mode="nearest") if smoothing else None}
    fwhm, dispersion = corrected(fwhm, dispersion)
    simulated = []
    for k in range(n_realizations):
        noisy = y + rng.standard_normal(len(y))*y_error
        values = profile_quantities(x, noisy, continuum, smoothing)
        simulated.append(values)
    s_flux, s_ew, s_centroid, s_dispersion, s_fwhm = np.array(simulated).T
    s_fwhm, s_dispersion = corrected(s_fwhm, s_dispersion)
    return {"flux": summarize_around(flux, s_flux),
            "equivalent_width_rest": summarize_around(ew/(1 + z_value), s_ew/(1 + z_draw)),
            "fwhm_rest": summarize_around(fwhm/(1 + z_value), s_fwhm/(1 + z_draw)),
            "dispersion_rest": summarize_around(dispersion/(1 + z_value), s_dispersion/(1 + z_draw)),
            "dispersion_velocity": summarize_around(SPEED_OF_LIGHT*dispersion/centroid, SPEED_OF_LIGHT*s_dispersion/s_centroid),
            "centroid": summarize_around(centroid, s_centroid), **measured}


# ======================================================================================================================================
# Black hole masses (single-epoch virial estimators)
# ======================================================================================================================================

def value_and_errors(quantity, unit=None):

    """
    (value, error_down, error_up) of an input given as a number, an astropy Quantity, or a list/array [value, error] or
    [value, error_down, error_up] (e.g. one row of the outputs of line_physics()).
    """

    if hasattr(quantity, "unit") and unit is not None:
        quantity = quantity.to(unit)
    values = np.atleast_1d(np.asarray(getattr(quantity, "value", quantity), dtype=float))
    if len(values) == 1:
        return float(values[0]), 0.0, 0.0
    if len(values) == 2:
        return float(values[0]), abs(float(values[1])), abs(float(values[1]))
    if len(values) == 3:
        return float(values[0]), abs(float(values[1])), abs(float(values[2]))
    raise ValueError(f"Expected a value, [value, error] or [value, error_down, error_up]; got {values}.")


def asymmetric_samples(value, down, up, n_samples, rng, positive=False):

    """Samples of a split normal distribution (value, error_down, error_up); optionally restricted to positive values."""

    normal = rng.standard_normal(n_samples)
    samples = value + normal*np.where(normal < 0, down, up)
    if positive:
        samples = np.where(samples > 0, samples, np.nan)
    return samples


def luminosity_distance_cm(redshift, H0, Om0):

    """Luminosity distance (cm) in a flat LambdaCDM cosmology (no radiation term, as in the calibrations of the virial estimators)."""

    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    return FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(np.asarray(redshift, dtype=float)).to(u.cm).value


def continuum_flux_density(wavelengths, continuum, redshift, rest_wavelength, half_width=25.0):

    """
    Observed continuum flux density (erg/cm2/s/A) at the rest wavelength 'rest_wavelength', averaged over +-half_width rest-frame Angstroms,
    or np.nan if this window is not fully covered by the spectrum.
    """

    centre = rest_wavelength*(1 + redshift)
    low, high = (rest_wavelength - half_width)*(1 + redshift), (rest_wavelength + half_width)*(1 + redshift)
    if low < np.nanmin(wavelengths) or high > np.nanmax(wavelengths):
        return np.nan, centre
    inside = (wavelengths >= low) & (wavelengths <= high)
    return float(np.nanmean(continuum[inside])), centre
