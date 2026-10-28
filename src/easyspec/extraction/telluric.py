"""
Correction of the telluric absorption bands (O2 and H2O absorption in the Earth's atmosphere).

Two sources of the telluric transmission are supported:

1. The standard star (default; the classical method of e.g. Wade & Horne 1988, ApJ 324, 411, and of the IRAF task "telluric"). In each
   telluric band, the transmission is the ratio between the observed spectrum of the star and its continuum, interpolated across the band
   from the regions on both sides. This works best with a hot star with no intrinsic features in the bands.
2. A transmission model computed by an external code (e.g. molecfit, ESO SkyCalc or TAPAS), given as a table of wavelength and
   transmission. It is convolved to the resolution of the spectra and rebinned to their wavelengths.

The transmission is scaled from the airmass of the reference (standard star or model) to the airmass of the target as

    T_target = T_reference ** ((X_target/X_reference) ** beta),

where beta ~ 0.55 for the strong (saturated) bands observed at low/moderate resolution (Wade & Horne 1988). Optionally, the scaling
exponent and a small wavelength shift can be fitted to each band of each target spectrum, by minimizing the residual absorption.
"""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Telluric bands (air wavelengths, Angstrom) and the molecule responsible for each one:
TELLURIC_BANDS = {"O2 gamma": [6270.0, 6330.0], "O2 B": [6855.0, 6960.0], "H2O 7200": [7160.0, 7340.0], "O2 A": [7585.0, 7705.0],
                  "H2O 8200": [8120.0, 8350.0], "H2O 9300": [8940.0, 9800.0]}

TELLURIC_MASK_BIT = 16   # Bit of the 1D mask: strong telluric absorption (transmission < 0.5), corrected but less reliable


def _robust_polyfit(x, y, weights, order, clip_sigma=3.0, max_iterations=5):

    """Weighted polynomial fit with iterative sigma clipping. Returns the polynomial (np.polynomial.Polynomial) or None."""

    use = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(use) < order + 2:
        return None
    polynomial = None
    for _ in range(max_iterations):
        polynomial = np.polynomial.Polynomial.fit(x[use], y[use], order, w=weights[use])
        residuals = (y - polynomial(x))*weights
        scale = 1.4826*np.median(np.abs(residuals[use]))
        new_use = use & (np.abs(residuals) <= clip_sigma*max(scale, 1e-30))
        if np.array_equal(new_use, use) or np.count_nonzero(new_use) < order + 2:
            break
        use = new_use
    return polynomial


def band_continuum(wavelength, flux, variance, band, window=100.0, order=1, other_bands=()):

    """
    Continuum across a telluric band, fitted to the regions [band[0] - window, band[0]] and [band[1], band[1] + window] (excluding other
    telluric bands).

    Returns
    -------
    continuum: array or None
        The continuum at all wavelengths (None if there are not enough points around the band).
    sides: int
        Number of sides of the band with data (0, 1 or 2). With only one side, the continuum is a constant.
    """

    wavelength, flux = np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)
    left = (wavelength >= band[0] - window) & (wavelength < band[0])
    right = (wavelength > band[1]) & (wavelength <= band[1] + window)
    for other in other_bands:
        inside_other = (wavelength >= other[0]) & (wavelength <= other[1])
        left &= ~inside_other
        right &= ~inside_other
    good = np.isfinite(flux)
    left &= good
    right &= good
    sides = int(np.count_nonzero(left) >= 5) + int(np.count_nonzero(right) >= 5)
    if sides == 0:
        return None, 0
    region = left | right
    weights = np.ones(len(wavelength))
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        weights = np.where(np.isfinite(variance) & (variance > 0), 1/np.sqrt(np.where(variance > 0, variance, 1.0)), 0.0)
    polynomial = _robust_polyfit(wavelength[region], flux[region], weights[region], order if sides == 2 else 0)
    if polynomial is None:
        return None, 0
    return polynomial(wavelength), sides


def transmission_from_standard(wavelength, flux, variance=None, bands=None, window=100.0, order=1):

    """
    Telluric transmission measured in the spectrum of a standard star: in each band, T = flux/continuum (T = 1 outside the bands).

    Parameters
    ----------
    wavelength, flux, variance: arrays
        Spectrum of the standard star (any units: the flux calibration cancels out in the ratio, since the sensitivity function is smooth
        across the telluric bands).
    bands: dict
        {name: [lambda_min, lambda_max]}. Default: TELLURIC_BANDS.
    window: float
        Width (Angstrom) of the regions on each side of a band used to fit the continuum.
    order: int
        Order of the polynomial continuum.

    Returns
    -------
    transmission: array
        Telluric transmission at each wavelength (1 outside the bands).
    transmission_variance: array
        Its variance (0 outside the bands).
    used_bands: dict
        The bands that could be measured, {name: [lambda_min, lambda_max]}.
    """

    bands = TELLURIC_BANDS if bands is None else bands
    wavelength, flux = np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)
    transmission = np.ones(len(wavelength))
    transmission_variance = np.zeros(len(wavelength))
    used_bands = {}
    for name, band in bands.items():
        inside = (wavelength >= band[0]) & (wavelength <= band[1])
        if np.count_nonzero(inside) < 3:
            continue
        others = [b for n, b in bands.items() if n != name]
        continuum, sides = band_continuum(wavelength, flux, variance, band, window=window, order=order, other_bands=others)
        if continuum is None:
            warnings.warn(f"Telluric band {name} ({band[0]:.0f}-{band[1]:.0f} A): not enough data around the band to measure the continuum of "
                          "the standard star. This band is not corrected.")
            continue
        if sides == 1:
            warnings.warn(f"Telluric band {name} ({band[0]:.0f}-{band[1]:.0f} A) is at the edge of the standard star spectrum: its continuum was "
                          "estimated from one side only (constant). Check the plots.")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux/continuum
            ratio_variance = variance/continuum**2 if variance is not None else np.zeros(len(flux))
        valid = inside & np.isfinite(ratio) & (continuum > 0)
        transmission[valid] = ratio[valid]
        transmission_variance[valid] = np.nan_to_num(ratio_variance[valid])
        used_bands[name] = band
    return transmission, transmission_variance, used_bands


def model_to_spectrum(model_wavelength, model_transmission, wavelength, fwhm):

    """
    Convolve a (high-resolution) transmission model with a Gaussian of the given FWHM (Angstrom) and interpolate it at 'wavelength'.
    """

    model_wavelength = np.asarray(model_wavelength, dtype=float)
    model_transmission = np.asarray(model_transmission, dtype=float)
    order = np.argsort(model_wavelength)
    model_wavelength, model_transmission = model_wavelength[order], model_transmission[order]
    step = np.median(np.diff(model_wavelength))
    grid = np.arange(model_wavelength[0], model_wavelength[-1], step)   # Uniform grid for the convolution
    on_grid = np.interp(grid, model_wavelength, model_transmission)
    sigma_pixels = fwhm/2.3548/step
    if sigma_pixels > 0.3:
        on_grid = gaussian_filter1d(on_grid, sigma_pixels, mode="nearest")
    # Average over the pixels of the spectrum (the model is usually much better sampled):
    edges = np.concatenate([[wavelength[0] - 0.5*(wavelength[1] - wavelength[0])], 0.5*(wavelength[1:] + wavelength[:-1]),
                            [wavelength[-1] + 0.5*(wavelength[-1] - wavelength[-2])]])
    cumulative = np.concatenate([[0.0], np.cumsum(on_grid)*step])
    grid_edges = np.concatenate([grid - step/2, [grid[-1] + step/2]])
    integral = np.interp(edges, grid_edges, cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        transmission = np.diff(integral)/np.diff(edges)
    outside = (wavelength < model_wavelength[0]) | (wavelength > model_wavelength[-1]) | ~np.isfinite(transmission)
    transmission[outside] = 1.0
    return transmission


def scale_transmission(transmission, airmass_ratio, beta=0.55):

    """T_target = T_reference ** (airmass_ratio ** beta), with airmass_ratio = X_target/X_reference."""

    transmission = np.clip(np.asarray(transmission, dtype=float), 1e-4, None)
    return transmission**(airmass_ratio**beta)


def fit_band_scaling(wavelength, flux, variance, transmission_function, band, power0, window=100.0, order=1, other_bands=(),
                     max_shift=1.5, powers=None):

    """
    Fit the exponent (power) and the wavelength shift of the transmission that minimize the residual absorption in one band of a target
    spectrum, i.e. that make the corrected spectrum closest to its continuum (interpolated across the band).

    Parameters
    ----------
    transmission_function: callable
        transmission_function(wavelength) -> transmission of the reference (not scaled).
    power0: float
        The exponent predicted from the airmasses ((X_target/X_reference)**beta).
    max_shift: float
        Maximum shift (Angstrom).

    Returns
    -------
    power, shift, improvement: floats
        The best exponent and shift, and the decrease of chi2 with respect to the prediction (power0, no shift). If the band cannot be
        fitted, (power0, 0.0, 0.0).
    """

    inside = (wavelength >= band[0]) & (wavelength <= band[1]) & np.isfinite(flux)
    continuum, sides = band_continuum(wavelength, flux, variance, band, window=window, order=order, other_bands=other_bands)
    if continuum is None or sides < 2 or np.count_nonzero(inside) < 5:
        return power0, 0.0, 0.0
    sigma = np.sqrt(variance[inside]) if variance is not None else np.full(np.count_nonzero(inside), np.nanstd(flux[inside]))
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(sigma))
    powers = np.geomspace(0.3, 3.0, 41)*power0 if powers is None else powers
    shifts = np.arange(-max_shift, max_shift + 1e-9, 0.1)

    def chi2(power, shift):
        corrected = flux[inside]/np.clip(transmission_function(wavelength[inside] - shift), 1e-3, None)**power
        return np.sum(((corrected - continuum[inside])/sigma)**2)

    reference = chi2(power0, 0.0)
    best = (reference, power0, 0.0)
    for shift in shifts:
        for power in powers:
            value = chi2(power, shift)
            if value < best[0]:
                best = (value, power, shift)
    return best[1], best[2], reference - best[0]
