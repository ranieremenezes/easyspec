"""
Auxiliary functions of easyspec.photometry for exoplanet transits: the transit light curve of a planet on a circular orbit in front of a
limb-darkened star, and its fit to a differential light curve (least squares + MCMC with emcee).

The occulted flux is computed by numerical integration over the stellar disk: for each radius r of the star, the length of the circle of
radius r covered by the planet times the intensity I(r) of the quadratic limb-darkening law,
    I(mu)/I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2,    mu = sqrt(1 - r^2).
The integral is done with Gauss-Legendre quadrature in a variable that clusters the nodes at the edges of the integration interval (where
the integrand has square-root behaviour); the result agrees with the analytic Mandel & Agol (2002) model to better than 1 ppm.
"""

import warnings

import numpy as np
from scipy.optimize import least_squares

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(48)


def limb_darkened_intensity(radius, limb_darkening):
    """Quadratic limb-darkening law, normalized to 1 at the centre of the disk."""
    first_coefficient, second_coefficient = limb_darkening
    mu = np.sqrt(np.clip(1 - radius**2, 0, None))
    return 1 - first_coefficient*(1 - mu) - second_coefficient*(1 - mu)**2


def _integral_over_interval(lower, upper, separation, planet_radius, limb_darkening):
    """Integral of I(r) x (length of the circle of radius r inside the planet disk) for r in [lower, upper] (arrays)."""
    # Variable change r = lower + (upper - lower) (1 - cos(pi t))/2: nodes cluster at both ends, where the integrand has sqrt behaviour
    t = 0.5*(_GAUSS_NODES + 1)
    radius = lower[:, None] + (upper - lower)[:, None]*0.5*(1 - np.cos(np.pi*t))[None, :]
    jacobian = (upper - lower)[:, None]*0.5*np.pi*np.sin(np.pi*t)[None, :]
    separation = separation[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (radius**2 + separation**2 - planet_radius**2)/(2*radius*separation)
    half_angle = np.arccos(np.clip(cosine, -1, 1))
    # Circles entirely inside the planet disk (planet covering the centre of the star)
    half_angle = np.where(radius < planet_radius - separation, np.pi, half_angle)
    integrand = limb_darkened_intensity(radius, limb_darkening)*2*radius*half_angle
    return np.sum(integrand*jacobian*0.5*_GAUSS_WEIGHTS[None, :], axis=1)


def occulted_fraction(separation, planet_radius, limb_darkening):
    """
    Fraction of the stellar flux blocked by a planet of radius planet_radius (in stellar radii) whose centre is at 'separation' (stellar
    radii) from the centre of the star.
    """
    separation = np.atleast_1d(np.asarray(separation, dtype=float))
    blocked = np.zeros_like(separation)
    total = np.pi*(1 - limb_darkening[0]/3 - limb_darkening[1]/6)
    overlapping = separation < 1 + planet_radius
    if planet_radius <= 0 or not np.any(overlapping):
        return blocked
    z = np.maximum(separation[overlapping], 1e-9)
    lower = np.maximum(z - planet_radius, 0.0)
    upper = np.minimum(z + planet_radius, 1.0)
    # Split at |z - p| when the planet covers the centre, so that each interval has a smooth integrand
    inner_edge = np.clip(planet_radius - z, 0.0, 1.0)
    covers_centre = planet_radius > z
    result = np.zeros_like(z)
    if np.any(covers_centre):
        # Fully covered disk of radius (p - z): integral of I(r) 2 pi r dr
        full_lower = np.zeros(np.count_nonzero(covers_centre))
        full_upper = inner_edge[covers_centre]
        t = 0.5*(_GAUSS_NODES + 1)
        radius = full_lower[:, None] + (full_upper - full_lower)[:, None]*t[None, :]
        result[covers_centre] += np.sum(limb_darkened_intensity(radius, limb_darkening)*2*np.pi*radius*0.5*_GAUSS_WEIGHTS[None, :],
                                        axis=1)*(full_upper - full_lower)
        lower = np.where(covers_centre, inner_edge, lower)
    valid = upper > lower
    if np.any(valid):
        result[valid] += _integral_over_interval(lower[valid], upper[valid], z[valid], planet_radius, limb_darkening)
    blocked[overlapping] = result/total
    return blocked


def sky_projected_separation(time, mid_transit_time, period, scaled_semi_major_axis, inclination_degrees):
    """Separation between the centres of the planet and the star (stellar radii) on a circular orbit, and whether the planet is in front."""
    phase = 2*np.pi*(np.asarray(time, dtype=float) - mid_transit_time)/period
    x = scaled_semi_major_axis*np.sin(phase)
    y = scaled_semi_major_axis*np.cos(phase)*np.cos(np.radians(inclination_degrees))
    in_front = np.cos(phase) > 0
    return np.hypot(x, y), in_front


def transit_light_curve(time, mid_transit_time, period, planet_radius, scaled_semi_major_axis, inclination_degrees, limb_darkening,
                        exposure_time=None, supersampling=5):
    """
    Relative flux of the star (1 out of transit). Times and period in days; planet_radius = Rp/Rs; scaled_semi_major_axis = a/Rs;
    exposure_time in days (if given, the model is averaged over the exposure with 'supersampling' points).
    """
    time = np.asarray(time, dtype=float)
    if exposure_time and supersampling > 1:
        offsets = ((np.arange(supersampling) + 0.5)/supersampling - 0.5)*exposure_time
        sampled_time = (time[:, None] + offsets[None, :]).ravel()
    else:
        sampled_time = time
    separation, in_front = sky_projected_separation(sampled_time, mid_transit_time, period, scaled_semi_major_axis, inclination_degrees)
    flux = np.ones_like(sampled_time)
    candidates = in_front & (separation < 1 + planet_radius)
    if np.any(candidates):
        flux[candidates] = 1 - occulted_fraction(separation[candidates], planet_radius, limb_darkening)
    if exposure_time and supersampling > 1:
        flux = flux.reshape(len(time), supersampling).mean(axis=1)
    return flux


def transit_duration(period, planet_radius, scaled_semi_major_axis, inclination_degrees):
    """Total duration T14 (same units as the period) of a transit on a circular orbit (NaN if there is no transit)."""
    impact_parameter = scaled_semi_major_axis*np.cos(np.radians(inclination_degrees))
    argument = np.sqrt(np.clip((1 + planet_radius)**2 - impact_parameter**2, 0, None))/(scaled_semi_major_axis
                                                                                        *np.sin(np.radians(inclination_degrees)))
    if argument <= 0:
        return np.nan
    return period/np.pi*np.arcsin(np.clip(argument, 0, 1))


# ======================================================================================================================================
# Fit
# ======================================================================================================================================

PARAMETER_NAMES = ("mid_transit_time", "planet_radius", "scaled_semi_major_axis", "impact_parameter", "limb_darkening_q1",
                   "limb_darkening_q2")


def limb_darkening_from_kipping(q1, q2):
    """Quadratic coefficients (u1, u2) from Kipping's (2013) parametrization (q1, q2 in [0, 1] sample all physical laws uniformly)."""
    return 2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)


def kipping_from_limb_darkening(u1, u2):
    """Inverse of limb_darkening_from_kipping()."""
    q1 = (u1 + u2)**2
    q2 = u1/(2*(u1 + u2)) if (u1 + u2) != 0 else 0.5
    return q1, q2


def baseline_design_matrix(time, airmass=None, polynomial_order=1, use_airmass=False, extra_regressors=None):
    """Columns of the multiplicative baseline: polynomial in (normalized) time, optional airmass and extra regressors (e.g. FWHM, sky)."""
    normalized_time = (time - np.mean(time))/max(np.ptp(time), 1e-12)
    columns = [normalized_time**power for power in range(polynomial_order + 1)]
    if use_airmass and airmass is not None:
        columns.append(np.asarray(airmass, dtype=float) - np.mean(airmass))
    for regressor in (extra_regressors or []):
        regressor = np.asarray(regressor, dtype=float)
        columns.append((regressor - np.mean(regressor))/max(np.std(regressor), 1e-12))
    return np.vstack(columns).T


class TransitModel:

    """
    Transit times a linear baseline. The baseline coefficients are solved by weighted linear least squares for each set of transit
    parameters (profile likelihood), so that only the transit parameters are sampled.
    """

    def __init__(self, time, flux, flux_error, period, design_matrix, exposure_time=None, fixed=None, priors=None, bounds=None):
        self.time, self.flux, self.flux_error = np.asarray(time), np.asarray(flux), np.asarray(flux_error)
        self.period = period
        self.design_matrix = design_matrix
        self.exposure_time = exposure_time
        self.fixed = dict(fixed or {})
        self.priors = dict(priors or {})                   # {name: (mean, sigma)} Gaussian priors
        self.free_names = [name for name in PARAMETER_NAMES if name not in self.fixed]
        self.bounds = bounds

    def full_parameters(self, free_values):
        parameters = dict(self.fixed)
        parameters.update(dict(zip(self.free_names, free_values)))
        return parameters

    def transit(self, parameters, time=None):
        time = self.time if time is None else time
        a = parameters["scaled_semi_major_axis"]
        inclination = np.degrees(np.arccos(np.clip(parameters["impact_parameter"]/a, -1, 1)))
        limb_darkening = limb_darkening_from_kipping(parameters["limb_darkening_q1"], parameters["limb_darkening_q2"])
        return transit_light_curve(time, parameters["mid_transit_time"], self.period, parameters["planet_radius"], a, inclination,
                                   limb_darkening, exposure_time=self.exposure_time if time is self.time else None)

    def baseline(self, transit_shape):
        weighted = self.design_matrix*transit_shape[:, None]/self.flux_error[:, None]
        coefficients = np.linalg.lstsq(weighted, self.flux/self.flux_error, rcond=None)[0]
        return self.design_matrix @ coefficients, coefficients

    def residuals(self, free_values):
        parameters = self.full_parameters(free_values)
        shape = self.transit(parameters)
        baseline, _ = self.baseline(shape)
        normalized = (self.flux - baseline*shape)/self.flux_error
        penalties = [(parameters[name] - mean)/sigma for name, (mean, sigma) in self.priors.items() if name in self.free_names]
        return np.r_[normalized, penalties]

    def log_probability(self, free_values):
        parameters = self.full_parameters(free_values)
        lower, upper = self.bounds
        if np.any(np.asarray(free_values) < lower) or np.any(np.asarray(free_values) > upper):
            return -np.inf
        if parameters["impact_parameter"] > 1 + parameters["planet_radius"]:
            return -np.inf
        return -0.5*np.sum(self.residuals(free_values)**2)


def red_noise_factor(residuals, bin_sizes=range(4, 31)):
    """Time-averaging beta factor (Pont et al. 2006; Winn et al. 2008): ratio of the scatter of binned residuals to white-noise expectation."""
    residuals = np.asarray(residuals)
    single_point_scatter = np.std(residuals)
    factors = []
    for size in bin_sizes:
        number_of_bins = len(residuals)//size
        if number_of_bins < 5:
            continue
        binned = residuals[:number_of_bins*size].reshape(number_of_bins, size).mean(axis=1)
        expected = single_point_scatter/np.sqrt(size)*np.sqrt(number_of_bins/(number_of_bins - 1))
        factors.append(np.std(binned)/expected)
    return max(1.0, float(np.median(factors))) if factors else 1.0


def multistart_least_squares(model, starting_points):
    """Least-squares fit from several starting points; returns the best."""
    best = None
    lower, upper = model.bounds
    for start in starting_points:
        start = np.clip(start, lower + 1e-9*(upper - lower), upper - 1e-9*(upper - lower))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = least_squares(model.residuals, start, bounds=(lower, upper), x_scale="jac")
        if best is None or fit.cost < best.cost:
            best = fit
    return best
