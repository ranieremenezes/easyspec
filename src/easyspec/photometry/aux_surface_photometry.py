"""
Auxiliary functions of easyspec.photometry for surface photometry of extended objects (globular clusters, galaxies): light centre,
surface-brightness profiles in elliptical annuli, isophote fitting (photutils.isophote), and King (1962) and Sersic profile fits.
"""

import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import gammaincinv


def light_center(data, mask=None, smoothing_sigma=10.0):
    """
    Centre of the light distribution (x, y) of an extended object: maximum of the image smoothed with a Gaussian by normalized convolution
    (masked pixels have zero weight, so masked stars do not create holes), refined by the intensity-weighted centroid of the smoothed image
    within one smoothing length.
    """
    weights = np.ones(data.shape) if mask is None else (~mask).astype(float)
    weights[~np.isfinite(data)] = 0.0
    filled = np.where(weights > 0, data, 0.0)
    smoothed = ndimage.gaussian_filter(filled*weights, smoothing_sigma)/np.maximum(ndimage.gaussian_filter(weights, smoothing_sigma), 1e-6)
    # Ignore the borders, where the smoothing is unreliable
    border = int(2*smoothing_sigma)
    inner = np.full_like(smoothed, -np.inf)
    inner[border:-border, border:-border] = smoothed[border:-border, border:-border]
    y_peak, x_peak = np.unravel_index(np.argmax(inner), smoothed.shape)
    radius = int(np.ceil(smoothing_sigma))
    window = (slice(max(y_peak - radius, 0), y_peak + radius + 1), slice(max(x_peak - radius, 0), x_peak + radius + 1))
    stamp = smoothed[window] - np.min(smoothed[window])
    y_grid, x_grid = np.mgrid[window]
    return float(np.sum(stamp*x_grid)/np.sum(stamp)), float(np.sum(stamp*y_grid)/np.sum(stamp))


def elliptical_radius(shape, center, ellipticity=0.0, position_angle_degrees=0.0):
    """
    Semi-major-axis distance of every pixel to the centre, for ellipses of the given ellipticity (1 - b/a) and position angle (degrees,
    counter-clockwise from the +x axis, the photutils.isophote convention).
    """
    y_grid, x_grid = np.indices(shape, dtype=float)
    angle = np.radians(position_angle_degrees)
    along_major = (x_grid - center[0])*np.cos(angle) + (y_grid - center[1])*np.sin(angle)
    along_minor = -(x_grid - center[0])*np.sin(angle) + (y_grid - center[1])*np.cos(angle)
    return np.hypot(along_major, along_minor/(1 - ellipticity))


def elliptical_radius_of_points(positions, center, ellipticity=0.0, position_angle_degrees=0.0):
    """Semi-major-axis distance of points (x, y) to the centre, for ellipses of the given ellipticity and position angle."""
    angle = np.radians(position_angle_degrees)
    along_major = (positions[:, 0] - center[0])*np.cos(angle) + (positions[:, 1] - center[1])*np.sin(angle)
    along_minor = -(positions[:, 0] - center[0])*np.sin(angle) + (positions[:, 1] - center[1])*np.cos(angle)
    return np.hypot(along_major, along_minor/(1 - ellipticity))


def annular_profile(data, variance, mask, center, radius_edges, ellipticity=0.0, position_angle_degrees=0.0, clip_sigma=3.0, fwhm=1.0,
                    star_positions=None, star_fluxes=None):
    """
    Robust mean intensity in elliptical annuli (sigma-clipped mean of the unmasked pixels, which rejects residuals of stars) and its
    uncertainty. The uncertainty is the larger of the propagated pixel variance and the empirical error of the mean: the scatter of the
    pixels divided by sqrt(N / N_PSF), where N_PSF = 1.13 FWHM^2 is the number of pixels per independent resolution element (the
    fluctuations of unresolved stars, which dominate in the inner parts of clusters and galaxies, are correlated over the PSF).
    If star_positions and star_fluxes are given (stars subtracted from the image), the summed flux of the stars whose centres fall in each
    annulus, divided by the area of the annulus, is added back: the result is the total surface brightness (resolved + unresolved light),
    with the Poisson noise of the number of stars added to the error.
    Returns a dict of arrays: radius (mean semi-major axis of the pixels used), intensity, error, number of pixels, fraction masked.
    """
    radius_map = elliptical_radius(data.shape, center, ellipticity, position_angle_degrees)
    usable = ~mask & np.isfinite(data)
    if variance is None:
        variance = np.full(data.shape, np.nan)
    output = {name: [] for name in ("radius", "inner_radius", "outer_radius", "intensity", "intensity_error", "number_of_pixels",
                                    "masked_fraction", "resolved_star_intensity")}
    if star_positions is not None:
        star_radius = elliptical_radius_of_points(star_positions, center, ellipticity, position_angle_degrees)
    for inner, outer in zip(radius_edges[:-1], radius_edges[1:]):
        in_annulus = (radius_map >= inner) & (radius_map < outer)
        total = np.count_nonzero(in_annulus)
        selected = in_annulus & usable
        number = np.count_nonzero(selected)
        if number < 5:
            continue
        values = data[selected]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clipped_mean, _, clipped_std = sigma_clipped_stats(values, sigma=clip_sigma, maxiters=10)
        kept = np.abs(values - clipped_mean) < clip_sigma*max(clipped_std, 1e-30)
        propagated = np.sqrt(np.nansum(variance[selected][kept]))/max(np.count_nonzero(kept), 1)
        number_kept = max(np.count_nonzero(kept), 1)
        empirical = clipped_std/np.sqrt(number_kept/min(1.13*fwhm**2, number_kept))
        output["radius"].append(float(np.mean(radius_map[selected][kept])))
        output["inner_radius"].append(float(inner))
        output["outer_radius"].append(float(outer))
        error = max(propagated, empirical)
        resolved = 0.0
        if star_positions is not None:
            in_ring = (star_radius >= inner) & (star_radius < outer)
            resolved = float(np.sum(star_fluxes[in_ring]))/total
            error = np.hypot(error, np.sqrt(np.sum(star_fluxes[in_ring]**2))/total)
        output["intensity"].append(float(clipped_mean + resolved))
        output["intensity_error"].append(float(error))
        output["resolved_star_intensity"].append(resolved)
        output["number_of_pixels"].append(int(np.count_nonzero(kept)))
        output["masked_fraction"].append(float(1 - number/total))
    return {name: np.array(values) for name, values in output.items()}


def logarithmic_radius_edges(minimum_radius, maximum_radius, number_of_annuli, minimum_width=1.0):
    """Annulus edges spaced logarithmically between minimum_radius and maximum_radius (each at least minimum_width pixels), starting at 0."""
    edges = [0.0]
    for radius in np.geomspace(max(minimum_radius, 0.5), maximum_radius, number_of_annuli):
        if radius - edges[-1] >= minimum_width:
            edges.append(float(radius))
    return np.array(edges)


def fit_isophotes(data, mask, center, initial_semi_major_axis, initial_ellipticity=0.1, initial_position_angle_degrees=0.0,
                  maximum_semi_major_axis=None, step=0.1, smoothing_size=None):
    """
    Isophote fitting with photutils.isophote.Ellipse (Jedrzejewski 1987). Masked pixels are ignored; with smoothing_size (pixels) the
    image is median filtered first (needed when the light is grainy, e.g. partially resolved objects). Returns the photutils IsophoteList
    (with sma, intens, int_err, eps, pa (radians), x0, y0, ...) or None if the fit fails.
    """
    from photutils.isophote import Ellipse, EllipseGeometry
    unusable = mask | ~np.isfinite(data)
    if smoothing_size and smoothing_size > 1:
        # The isophote fit needs a smooth image: masked pixels are filled with the global median and the image is median filtered
        filled = np.where(unusable, np.median(data[~unusable]), data)
        data = ndimage.median_filter(filled, size=int(smoothing_size))
        unusable = ndimage.binary_erosion(unusable, iterations=int(smoothing_size)//2)
    masked = np.ma.masked_array(data, mask=unusable)
    geometry = EllipseGeometry(x0=center[0], y0=center[1], sma=initial_semi_major_axis, eps=initial_ellipticity,
                               pa=np.radians(initial_position_angle_degrees))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            isophotes = Ellipse(masked, geometry).fit_image(sma0=initial_semi_major_axis, maxsma=maximum_semi_major_axis, step=step,
                                                            nclip=3, sclip=3.0, fflag=0.5)
        except Exception:
            return None
    return isophotes if len(isophotes) else None


# ======================================================================================================================================
# Profile models
# ======================================================================================================================================

def king_profile(radius, central_intensity, core_radius, tidal_radius):
    """King (1962) surface-brightness profile; zero beyond the tidal radius. central_intensity is the intensity at r = 0."""
    constant = 1/np.sqrt(1 + (tidal_radius/core_radius)**2)
    shape = np.clip(1/np.sqrt(1 + (np.asarray(radius)/core_radius)**2) - constant, 0, None)**2/(1 - constant)**2
    return central_intensity*shape


def sersic_b(sersic_index):
    """b_n such that the effective radius contains half of the light (exact, from the incomplete gamma function)."""
    return gammaincinv(2*sersic_index, 0.5)


def sersic_profile(radius, effective_intensity, effective_radius, sersic_index):
    """Sersic profile I(r) = I_e exp(-b_n [(r/r_e)^(1/n) - 1])."""
    return effective_intensity*np.exp(-sersic_b(sersic_index)*((np.asarray(radius)/effective_radius)**(1/sersic_index) - 1))


def fit_profile(radius, intensity, intensity_error, model_function, parameter_names, initial_values, lower_bounds, upper_bounds,
                fixed_values=None, fit_sky=True, initial_sky=0.0):

    """
    Weighted least-squares fit of model_function(radius, *parameters) + sky to a surface-brightness profile. Parameters can be fixed with
    fixed_values = {name: value}. The errors of the parameters come from the covariance matrix scaled by the reduced chi2 (the profile
    errors do not include all correlations, e.g. those of unresolved stars). Returns a dict.
    """

    fixed_values = dict(fixed_values or {})
    free_names = [name for name in parameter_names if name not in fixed_values]
    if fit_sky and "sky" not in fixed_values:
        free_names.append("sky")
    initial = dict(zip(parameter_names, initial_values), sky=initial_sky)
    lower = dict(zip(parameter_names, lower_bounds), sky=-np.inf)
    upper = dict(zip(parameter_names, upper_bounds), sky=np.inf)

    def unpack(free_values):
        values = dict(fixed_values)
        values.update(zip(free_names, free_values))
        values.setdefault("sky", 0.0)
        return values

    def residuals(free_values):
        values = unpack(free_values)
        model = model_function(radius, *[values[name] for name in parameter_names]) + values["sky"]
        return (intensity - model)/intensity_error

    start = np.array([initial[name] for name in free_names], dtype=float)
    bounds = (np.array([lower[name] for name in free_names]), np.array([upper[name] for name in free_names]))
    margin_low = np.where(np.isfinite(bounds[0]), 1e-6*np.maximum(np.abs(bounds[0]), 1e-12), 0.0)
    margin_high = np.where(np.isfinite(bounds[1]), 1e-6*np.maximum(np.abs(bounds[1]), 1e-12), 0.0)
    start = np.clip(start, bounds[0] + margin_low, bounds[1] - margin_high)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = least_squares(residuals, start, bounds=bounds, x_scale="jac", max_nfev=5000)
    degrees_of_freedom = max(len(radius) - len(free_names), 1)
    reduced_chi2 = float(2*fit.cost/degrees_of_freedom)
    try:
        covariance = np.linalg.inv(fit.jac.T @ fit.jac)*max(reduced_chi2, 1.0)
        uncertainties = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        covariance, uncertainties = None, np.full(len(free_names), np.nan)
    values = unpack(fit.x)
    errors = {name: 0.0 for name in values}
    errors.update(dict(zip(free_names, uncertainties)))
    return {"values": values, "errors": errors, "free_parameters": free_names, "reduced_chi2": reduced_chi2, "covariance": covariance,
            "success": bool(fit.success), "model": lambda r: model_function(r, *[values[name] for name in parameter_names]) + values["sky"]}
