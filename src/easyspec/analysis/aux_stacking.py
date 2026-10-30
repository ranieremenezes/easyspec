"""
Auxiliary functions of easyspec.analysis for stacking (co-adding) calibrated 1D spectra: flux-conserving resampling onto a
common grid (with propagation of the uncertainties and of the data quality masks), relative normalization and robust combination.
"""

import numpy as np
from scipy import sparse

# Bits of the 1D data quality mask that exclude a pixel from the stack (see easyspec.extraction.optimal_extraction.SPECTRUM_MASK_BITS):
BAD_SPECTRUM_BITS = 1 | 4 | 8
NO_DATA_BIT = 4           # Bit set in the stacked mask where no spectrum contributes
STACK_METHODS = ("median", "mean", "weighted", "sigma_clip")


def pixel_edges(centers):

    """Edges of the pixels of a (strictly increasing) wavelength array: midpoints between centers, extrapolated at the ends."""

    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        raise ValueError("A spectrum needs at least two pixels.")
    middle = 0.5*(centers[1:] + centers[:-1])
    return np.concatenate([[centers[0] - (middle[0] - centers[0])], middle, [centers[-1] + (centers[-1] - middle[-1])]])


def overlap_matrix(old_edges, new_edges):

    """
    Sparse matrix O (n_new x n_old) with the length of the overlap between each new and each old pixel, so that the
    flux-conserving resampling of a flux density f is (O @ f) / (O @ 1).
    """

    n_old, n_new = len(old_edges) - 1, len(new_edges) - 1
    first = np.clip(np.searchsorted(new_edges, old_edges[:-1], side="right") - 1, 0, n_new - 1)
    last = np.clip(np.searchsorted(new_edges, old_edges[1:], side="left") - 1, 0, n_new - 1)
    counts = np.maximum(last - first + 1, 0)
    old_index = np.repeat(np.arange(n_old), counts)
    new_index = np.repeat(first, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
    overlap = (np.minimum(new_edges[new_index + 1], old_edges[old_index + 1])
               - np.maximum(new_edges[new_index], old_edges[old_index]))
    keep = overlap > 0
    return sparse.csr_matrix((overlap[keep], (new_index[keep], old_index[keep])), shape=(n_new, n_old))


def resample_spectrum(wavelength, flux, new_wavelength, flux_err=None, calibration_err=None, mask=None, min_coverage=0.5):

    """
    Flux-conserving resampling of a spectrum (flux density) onto a new wavelength grid.

    Each new pixel receives the average of the old pixels weighted by their overlap with it. Bad pixels (non-finite flux or
    uncertainty, or mask bits in BAD_SPECTRUM_BITS) are left out; a new pixel is NaN if less than 'min_coverage' of its width is
    covered by good data, or if it is outside the spectrum. Uncertainties are propagated assuming independent old pixels (the
    resampled ones are correlated if the new grid is finer than the old one); the calibration uncertainty, which is correlated
    between neighbouring pixels, is averaged like the flux. The mask of a new pixel is the bitwise OR of the old pixels that overlap it, except for the
    bad-pixel bits, which are kept only where the new pixel has no good data.

    Returns
    -------
    new_flux, new_err, new_calibration_err, new_mask: numpy.ndarray
        new_err and new_calibration_err are None if the corresponding inputs are None.
    """

    wavelength, flux = np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)
    good = np.isfinite(wavelength) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        good &= np.isfinite(flux_err) & (flux_err > 0)
    if mask is not None:
        mask = np.asarray(mask).astype(np.int64)
        good &= (mask & BAD_SPECTRUM_BITS) == 0
    finite = np.isfinite(wavelength)
    order = np.argsort(wavelength[finite])
    select = np.flatnonzero(finite)[order]
    wavelength, flux, good = wavelength[select], flux[select], good[select]
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("The wavelengths of a spectrum must be distinct.")

    old_edges, new_edges = pixel_edges(wavelength), pixel_edges(new_wavelength)
    matrix = overlap_matrix(old_edges, new_edges)
    weights = good.astype(float)
    covered = matrix @ weights
    width = np.diff(new_edges)
    valid = covered >= min_coverage*width
    with np.errstate(invalid="ignore", divide="ignore"):
        new_flux = np.where(valid, matrix @ np.where(good, flux, 0.0) / covered, np.nan)
        new_err = None
        if flux_err is not None:
            variance = (matrix.multiply(matrix)) @ np.where(good, flux_err[select]**2, 0.0)
            new_err = np.where(valid, np.sqrt(variance) / covered, np.nan)
        new_calibration = None
        if calibration_err is not None:
            calibration_err = np.asarray(calibration_err, dtype=float)[select]
            usable = good & np.isfinite(calibration_err)
            new_calibration = np.where(valid, matrix @ np.where(usable, calibration_err, 0.0) / covered, np.nan)

    new_mask = np.zeros(len(width), dtype=np.int64)
    if mask is not None:
        mask = mask[select]
        touches = matrix.copy()
        touches.data[:] = 1.0
        for bit in (1 << b for b in range(int(mask.max()).bit_length() if mask.size and mask.max() > 0 else 0)):
            has_bit = ((mask & bit) > 0).astype(float)
            if has_bit.any():
                new_mask[(touches @ has_bit) > 0] |= bit
    new_mask[valid] &= ~BAD_SPECTRUM_BITS   # Bad old pixels were left out, so they only flag new pixels without good data
    new_mask[~valid] |= NO_DATA_BIT
    return new_flux, new_err, new_calibration, new_mask


def common_grid(wavelength_list, over_resolution_factor=1, wavelength_range=None):

    """
    Linear wavelength grid covering all spectra (or 'wavelength_range'), with a step equal to the finest median pixel size among
    the spectra divided by 'over_resolution_factor'.
    """

    if over_resolution_factor <= 0:
        raise ValueError("'over_resolution_factor' must be positive.")
    steps = [np.nanmedian(np.diff(np.sort(w[np.isfinite(w)]))) for w in wavelength_list]
    step = min(steps)/over_resolution_factor
    if wavelength_range is None:
        lower = min(np.nanmin(w) for w in wavelength_list)
        upper = max(np.nanmax(w) for w in wavelength_list)
    else:
        lower, upper = sorted(float(v) for v in wavelength_range)
    n_points = int(np.floor((upper - lower)/step)) + 1
    if n_points > 2_000_000:
        raise ValueError(f"The common grid would have {n_points} points. Reduce 'over_resolution_factor' or give 'wavelength_range'.")
    return lower + step*np.arange(n_points)


def scale_factors(flux_matrix, error_matrix, normalize, reference, window_mask):

    """
    Multiplicative factors that bring each (resampled) spectrum to a common flux level.

    normalize = "reference": each spectrum is scaled to the reference spectrum (index 'reference') using the pixels inside
    'window_mask' where both have data; "unity": each spectrum is divided by its median inside 'window_mask'. The factor is the
    ratio of the medians, which is robust against lines and outliers.
    """

    n_spectra = flux_matrix.shape[0]
    factors = np.ones(n_spectra)
    for i in range(n_spectra):
        if normalize == "unity":
            use = window_mask & np.isfinite(flux_matrix[i])
            if np.count_nonzero(use) < 5:
                raise ValueError(f"Spectrum {i} has fewer than 5 good pixels in the normalization window.")
            level = np.median(flux_matrix[i, use])
            if level <= 0:
                raise ValueError(f"Spectrum {i} has a non-positive median flux in the normalization window; choose another window.")
            factors[i] = 1.0/level
        elif normalize == "reference" and i != reference:
            use = window_mask & np.isfinite(flux_matrix[i]) & np.isfinite(flux_matrix[reference])
            if np.count_nonzero(use) < 5:
                raise ValueError(f"Spectrum {i} overlaps the reference spectrum in fewer than 5 good pixels of the normalization "
                                 "window, so it cannot be scaled to it. Use normalize='unity' or another 'normalization_window'.")
            level, level_ref = np.median(flux_matrix[i, use]), np.median(flux_matrix[reference, use])
            if level <= 0 or level_ref <= 0:
                raise ValueError(f"Non-positive median flux in the normalization window (spectrum {i} or the reference).")
            factors[i] = level_ref/level
    return factors


def combine(flux_matrix, error_matrix=None, method="median", clip_sigma=3.0, max_iterations=5):

    """
    Combine resampled spectra (rows of flux_matrix, NaN = no data) pixel by pixel.

    Methods: "median"; "mean"; "weighted" (inverse-variance weighted mean); "sigma_clip" (weighted mean, or mean without
    uncertainties, after iteratively rejecting points more than clip_sigma away from the median, in units of their uncertainty or,
    without uncertainties, of the robust scatter; needs at least 3 spectra in the pixel).

    Returns
    -------
    flux, error, scatter_error, n_used, used:
        error is the propagated uncertainty (None without error_matrix); scatter_error is the standard error from the dispersion of the
        spectra (NaN where fewer than 2 spectra contribute); n_used is the number of spectra used in each pixel and 'used' the boolean
        matrix of the points kept.
    """

    if method not in STACK_METHODS:
        raise ValueError(f"Invalid stacking method '{method}'. Options are {STACK_METHODS}.")
    flux_matrix = np.asarray(flux_matrix, dtype=float)
    used = np.isfinite(flux_matrix)
    if error_matrix is not None:
        used &= np.isfinite(error_matrix) & (error_matrix > 0)
    if method == "weighted" and error_matrix is None:
        raise ValueError("method='weighted' needs flux uncertainties for all spectra.")

    if method == "sigma_clip":
        for _ in range(max_iterations):
            data = np.where(used, flux_matrix, np.nan)
            enough = np.count_nonzero(used, axis=0) >= 3
            with np.errstate(invalid="ignore"), _quiet():
                centre = np.nanmedian(data, axis=0)
                if error_matrix is not None:
                    deviation = np.abs(data - centre)/error_matrix
                else:
                    scatter = 1.4826*np.nanmedian(np.abs(data - centre), axis=0)
                    deviation = np.abs(data - centre)/np.where(scatter > 0, scatter, np.inf)
            reject = used & enough & (deviation > clip_sigma)
            if not reject.any():
                break
            used &= ~reject

    data = np.where(used, flux_matrix, np.nan)
    n_used = np.count_nonzero(used, axis=0)
    has = n_used > 0
    flux = np.full(flux_matrix.shape[1], np.nan)
    error = None if error_matrix is None else np.full(flux_matrix.shape[1], np.nan)
    errors = None if error_matrix is None else np.where(used, error_matrix, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"), _quiet():
        if method == "median":
            flux[has] = np.nanmedian(data[:, has], axis=0)
            if errors is not None:
                # Uncertainty of the median of N Gaussian points with different sigmas (large-N limit): var = (pi/2)*N/(sum 1/sigma)^2,
                # which is (pi/2)*sigma^2/N for equal sigmas (slightly conservative for few spectra). With 1 or 2 spectra the median is the mean.
                mean_error = np.sqrt(np.nansum(errors[:, has]**2, axis=0))/n_used[has]
                median_error = np.sqrt(np.pi/2*n_used[has])/np.nansum(1.0/errors[:, has], axis=0)
                error[has] = np.where(n_used[has] > 2, median_error, mean_error)
        elif method == "mean" or (method == "sigma_clip" and errors is None):
            flux[has] = np.nanmean(data[:, has], axis=0)
            if errors is not None:
                error[has] = np.sqrt(np.nansum(errors[:, has]**2, axis=0))/n_used[has]
        else:   # weighted, or sigma_clip with uncertainties
            weights = np.where(used, 1.0/error_matrix**2, 0.0)
            total = weights.sum(axis=0)
            flux[has] = np.nansum(np.where(used, flux_matrix, 0.0)[:, has]*weights[:, has], axis=0)/total[has]
            error[has] = 1.0/np.sqrt(total[has])
        scatter_error = np.full(flux_matrix.shape[1], np.nan)
        several = n_used > 1
        scatter_error[several] = np.nanstd(data[:, several], axis=0, ddof=1)/np.sqrt(n_used[several])
    return flux, error, scatter_error, n_used, used


def consistency_chi2(flux_matrix, error_matrix, stacked_flux, used):

    """
    Reduced chi2 of each spectrum around the inverse-variance weighted mean of all spectra (pixels with at least two spectra). The
    residual of spectrum i is divided by sqrt(sigma_i^2 - sigma_mean^2), its expected scatter around a mean that includes it, so the
    value (from the median absolute deviation, robust against a few outliers) is about 1 if the spectra agree within their uncertainties (larger values: variability, calibration differences or
    underestimated errors). 'stacked_flux' is not used and kept for compatibility.
    """

    weights = np.where(used, 1.0/error_matrix**2, 0.0)
    total = weights.sum(axis=0)
    several = np.count_nonzero(used, axis=0) > 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(several, np.sum(np.where(used, flux_matrix, 0.0)*weights, axis=0)/total, np.nan)
        mean_variance = 1.0/total
    values = []
    for i in range(flux_matrix.shape[0]):
        use = used[i] & several
        if np.count_nonzero(use) < 10:
            values.append(np.nan)
            continue
        expected = error_matrix[i, use]**2 - mean_variance[use]
        residual = (flux_matrix[i, use] - mean[use])/np.sqrt(np.maximum(expected, 1e-12*error_matrix[i, use]**2))
        # Robust (MAD-based) estimate, so that a few outliers (e.g. a cosmic ray) do not dominate:
        values.append(float((1.4826*np.median(np.abs(residual - np.median(residual))))**2))
    return np.array(values)


class _quiet:

    """Context manager that silences the RuntimeWarnings of numpy nan-functions on all-NaN slices."""

    def __enter__(self):
        import warnings
        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        return self

    def __exit__(self, *args):
        return self._context.__exit__(*args)
