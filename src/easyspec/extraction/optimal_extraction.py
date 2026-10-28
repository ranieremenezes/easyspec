"""
Auxiliary functions for the spectral extraction of easyspec (used by extraction.extracting()).

They implement:

- the geometry of the extraction aperture and of the sky regions, which follow the (curved/tilted) trace with sub-pixel precision;
- a sky (background) fit in each column, whose uncertainty is propagated to the extracted spectrum;
- an empirical spatial profile, measured from the data themselves (no Gaussian assumption), which may vary slowly along the dispersion axis;
- the optimal extraction of Horne (1986, PASP 98, 609), with a variance model and the rejection of cosmic rays and bad pixels;
- a simple (tophat / boxcar) extraction over the same aperture, corrected for masked pixels with the spatial profile;
- the sub-pixel centroiding of the peaks of the lamp spectrum.

All arrays have the shape (n_rows, n_columns) of a horizontal band of the image around the trace, with the dispersion along the x axis.
"""

import warnings
import numpy as np
from scipy import interpolate, optimize

# Bits of the data quality mask of the extracted 1D spectra:
SPECTRUM_MASK_BITS = {1: "a large fraction of the spatial profile was masked or rejected (cosmic rays, bad pixels)",
                      2: "flat field not applied to some pixels of the aperture",
                      4: "no valid extraction (no good pixels, or non-finite result)",
                      8: "the aperture or the sky region falls partly outside the image"}

# Bits of the 2D data quality mask (see easyspec.data_io.MASK_BITS) that exclude a pixel from the extraction:
BAD_PIXEL_BITS = 1 | 4 | 8


def aperture_coverage(dy, half_width):

    """
    Fraction of each pixel (of size 1, centered at a distance 'dy' from the trace) that lies inside the aperture [-half_width, +half_width].
    Pixels at the edges of the aperture are therefore included with fractional weights, so that the aperture has exactly the same width in
    every column, whatever the sub-pixel position of the trace.
    """

    return np.clip(np.minimum(dy + 0.5, half_width) - np.maximum(dy - 0.5, -half_width), 0.0, 1.0)


def _polynomial_design(dy, order):

    """Design matrix of a polynomial of the given order in dy: shape dy.shape + (order + 1,)."""

    return np.stack([dy**k for k in range(order + 1)], axis=-1)


def fit_sky(data, variance, good, dy, sky_region, order=1, clip_sigma=3.0, max_iterations=5, gain_factor=None):

    """
    Fit, in each column, a polynomial of the given order in dy (the distance to the trace) to the sky pixels, with iterative sigma clipping.

    Parameters
    ----------
    data, variance: arrays (n_rows, n_columns)
        Data and variance of the band. 'variance' may be None (then the noise is estimated from the scatter of the sky pixels).
    good: boolean array
        Pixels that may be used (not masked).
    dy: array
        Distance (in pixels) of each pixel center to the trace.
    sky_region: boolean array
        Pixels of the sky region(s).
    order: int
        Order of the polynomial (0 = constant, 1 = linear gradient along the slit, ...).
    gain_factor:
        Optional (see estimate_variance_model). If given, after the first iteration the weights are computed from the variance of the sky
        model, model_variance(variance, data, sky, gain_factor), instead of the variance of the individual pixels. Weights computed from noisy
        variances would bias the sky low (pixels that fluctuated downwards get larger weights), which biases faint spectra high.

    Returns
    -------
    sky: array (n_rows, n_columns)
        Sky model evaluated at every pixel of the band.
    sky_variance: array (n_rows, n_columns)
        Variance of the sky model at every pixel (from the covariance of the fit).
    coefficients_covariance: array (n_columns, order+1, order+1)
        Covariance matrix of the polynomial coefficients in each column (needed to propagate the correlated sky error to the spectrum).
    column_ok: boolean array (n_columns,)
        Columns where the fit was possible.
    used: boolean array (n_rows, n_columns)
        Sky pixels used in the final fit (after the clipping).
    """

    n_parameters = order + 1
    design = _polynomial_design(dy, order)                                  # (ny, nx, k)
    candidates = good & sky_region & np.isfinite(data)
    if variance is not None:
        candidates &= np.isfinite(variance) & (variance > 0)
    used = candidates.copy()
    eye = np.eye(n_parameters)
    fit_variance = variance
    for iteration in range(max_iterations + 1):
        if fit_variance is not None:
            weight = np.where(used, 1.0/np.where(used, fit_variance, 1.0), 0.0)
        else:
            weight = used.astype(float)
        normal = np.einsum("yx,yxk,yxl->xkl", weight, design, design)       # (nx, k, k)
        rhs = np.einsum("yx,yxk,yx->xk", weight, design, np.where(used, data, 0.0))
        n_used = used.sum(axis=0)
        column_ok = n_used >= n_parameters + 2
        normal_safe = np.where(column_ok[:, None, None], normal, eye)
        with np.errstate(all="ignore"):
            coefficients = np.linalg.solve(normal_safe, np.where(column_ok[:, None], rhs, 0.0)[..., None])[..., 0]
            covariance = np.linalg.inv(normal_safe)
        sky = np.einsum("yxk,xk->yx", design, coefficients)
        residual = data - sky
        if gain_factor is not None and variance is not None:
            fit_variance = np.where(np.isfinite(sky), model_variance(variance, data, sky, gain_factor), variance)
        if fit_variance is not None:
            normalized = residual/np.sqrt(np.where(fit_variance > 0, fit_variance, np.inf))
            scale = np.ones(data.shape[1])
        else:
            # Without a variance, the noise of each column is estimated from the scatter of its sky pixels:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scale = 1.4826*np.nanmedian(np.abs(np.where(used, residual, np.nan)), axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.inf)
            normalized = residual/scale[None, :]
        # The clipping is always applied to all the candidate pixels (not cumulatively), so that pixels wrongly clipped in an early
        # iteration (e.g. when a cosmic ray pulled the first fit away) come back once the outlier is removed:
        new_used = candidates & (np.abs(normalized) <= clip_sigma)
        if np.array_equal(new_used, used) and (iteration > 0 or gain_factor is None):
            break
        used = new_used

    if variance is None:
        # Unweighted fit: covariance = (A^T A)^-1 * sigma^2, with sigma^2 estimated from the residuals of the sky pixels
        dof = np.maximum(used.sum(axis=0) - n_parameters, 1)
        sigma2 = np.sum(np.where(used, residual**2, 0.0), axis=0)/dof
        covariance = covariance*sigma2[:, None, None]
    covariance = np.where(column_ok[:, None, None], covariance, np.nan)
    sky_variance = np.einsum("yxk,xkl,yxl->yx", design, covariance, design)
    sky = np.where(column_ok[None, :], sky, np.nan)
    return sky, sky_variance, covariance, column_ok, used


def estimate_variance_model(data, variance, good):

    """
    Measure how the variance of each pixel depends on its own (noisy) value: b = d(variance)/d(data), so that the variance can be recomputed
    from the model of the data, V_model = variance + b*(model - data) (Horne's variance model).

    Why: easyspec.cleaning (like most pipelines) computes the Poisson part of the variance from the data themselves, so pixels that
    fluctuated upwards get larger variances, i.e. lower weights in the optimal extraction, which biases the extracted flux downwards (and the
    sky upwards). Replacing the data by the model removes this bias, while keeping all the other terms of the variance (read noise, noise of
    the calibration masters, flat-field noise) unchanged.

    b is measured from the differences between neighbouring rows (which remove the smooth signal and leave the noise): if the variance was
    computed from the data, these differences are correlated with those of the data, with slope b (= 1 for a single frame in electrons,
    smaller for combined or flat-fielded frames). If the variance does not depend on the noisy data (e.g. it was already computed from a
    model), b ~ 0 and the variance is used as given.

    Returns b (or None if it cannot be measured).
    """

    selection = good[1:] & good[:-1] & np.isfinite(data[1:]) & np.isfinite(data[:-1]) & np.isfinite(variance[1:]) & np.isfinite(variance[:-1])
    if np.count_nonzero(selection) < 100:
        return None
    d_data = (data[1:] - data[:-1])[selection]
    d_variance = (variance[1:] - variance[:-1])[selection]
    # Robust (clipped) least-squares slope through the origin:
    keep = np.ones(len(d_data), dtype=bool)
    slope = 0.0
    for _ in range(5):
        denominator = np.sum(d_data[keep]**2)
        if denominator <= 0:
            return None
        slope = np.sum(d_data[keep]*d_variance[keep])/denominator
        residual = d_variance - slope*d_data
        scale = 1.4826*np.median(np.abs(residual[keep]))
        new_keep = np.abs(residual) <= 5*max(scale, 1e-12) + 1e-12
        data_scale = 1.4826*np.median(np.abs(d_data))
        new_keep &= np.abs(d_data) <= 10*data_scale   # cosmic rays and strong features out
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if not np.isfinite(slope):
        return None
    return float(np.clip(slope, 0.0, None))


def model_variance(variance, data, model, gain_factor):

    """Variance recomputed from the model: variance + b*(model - data), never below 5% of the given variance (see estimate_variance_model)."""

    if gain_factor is None or gain_factor == 0:
        return variance
    return np.maximum(variance + gain_factor*(model - data), 0.05*variance)


class EmpiricalProfile:

    """
    Empirical spatial profile P(dy, x) of a spectrum: the fraction of the flux falling in a pixel centered at a distance dy from the trace.

    Because the trace crosses the pixel rows with different sub-pixel offsets in different columns, the normalized pixel values of many
    columns sample the profile at many values of dy, so that a smooth function of dy (a cubic B-spline) can be fitted to them with sub-pixel
    resolution. To allow the profile to change slowly along the dispersion axis (e.g. because of the wavelength dependence of the seeing),
    each B-spline coefficient is a polynomial of order 'x_order' in x:

        P(dy, x) = sum_j sum_m c_jm B_j(dy) L_m(x),

    where L_m are Legendre polynomials in the normalized x coordinate. All columns are fitted together (weighted least squares with iterative
    sigma clipping), so that even faint spectra constrain the variation of the profile without being split into noisy blocks.
    """

    def __init__(self, dy, values, weights, x_columns, x_order, half_width, nx, clip_sigma=4.0, max_iterations=5):

        self.half_width = half_width
        self.x_order = int(x_order)
        self.nx = nx
        limit = half_width + 0.5
        usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (np.abs(dy) <= limit)
        self.fwhm = self._estimate_fwhm(dy[usable], values[usable], weights[usable], limit)
        self.knots = self._knots(self.fwhm, limit)
        self.n_basis = len(self.knots) - 4

        rows_index, columns_index = np.nonzero(usable)
        # Range of x covered by the data: outside it, the profile is not extrapolated (the profile at the nearest edge is used)
        self.x_range = (float(x_columns[columns_index].min()), float(x_columns[columns_index].max())) if len(columns_index) else (0.0, nx - 1.0)
        if len(rows_index) < 3*self.n_basis*(self.x_order + 1):
            raise RuntimeError("Too few significant pixels to measure the spatial profile.")
        d = dy[usable]
        v = values[usable]
        w = weights[usable]
        u = self._x_normalized(x_columns[columns_index])
        design = self._design(d, u)                                   # (n_points, n_basis*(x_order+1))
        keep = np.ones(len(v), dtype=bool)
        coefficients = None
        for _ in range(max_iterations + 1):
            A = design[keep]*w[keep, None]
            try:
                coefficients, *_ = np.linalg.lstsq(A, v[keep]*w[keep], rcond=None)
            except np.linalg.LinAlgError:
                break
            normalized = (v - design @ coefficients)*w
            scale = 1.4826*np.median(np.abs(normalized[keep]))
            new_keep = np.abs(normalized) <= clip_sigma*max(scale, 1e-12)
            if np.array_equal(new_keep, keep) or np.count_nonzero(new_keep) < 2*design.shape[1]:
                break
            keep = new_keep
        if coefficients is None or not np.all(np.isfinite(coefficients)):
            raise RuntimeError("The spatial profile could not be fitted.")
        self.coefficients = coefficients.reshape(self.n_basis, self.x_order + 1)
        self.n_rejected = int(np.count_nonzero(~keep))

    @staticmethod
    def _estimate_fwhm(dy, values, weights, limit):

        """FWHM of the profile, from its weighted average in bins of 0.5 pixels (used only to place the knots of the B-spline)."""

        edges = np.arange(-limit, limit + 0.25, 0.5)
        index = np.digitize(dy, edges) - 1
        n = len(edges) - 1
        valid = (index >= 0) & (index < n)
        sum_w = np.bincount(index[valid], weights=weights[valid], minlength=n)
        sum_wv = np.bincount(index[valid], weights=(weights*values)[valid], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(sum_w > 0, sum_wv/sum_w, np.nan)
        centers = 0.5*(edges[:-1] + edges[1:])
        if not np.any(np.isfinite(mean)) or np.nanmax(mean) <= 0:
            return 4.0
        peak = np.nanmax(mean)
        above = np.isfinite(mean) & (mean >= 0.5*peak)
        width = np.ptp(centers[above]) + 0.5 if np.any(above) else 4.0
        return float(np.clip(width, 1.0, 2*limit))

    @staticmethod
    def _knots(fwhm, limit):

        """
        Knots of the cubic B-spline in dy: dense in the core of the profile (every FWHM/3 within +-1.5 FWHM), and progressively sparser in
        the wings, where the profile is smooth and the data are noisy. Using only the necessary number of free parameters keeps the noise
        of the measured profile (and the small bias it causes in the optimal extraction) low.
        """

        step = max(fwhm/3.0, 0.4)
        positions = [0.0]
        position, spacing = step, step
        while position < limit - 0.5*spacing:
            positions.append(position)
            if position >= 1.5*fwhm:
                spacing *= 1.5
            position += spacing
        positions = np.array(positions)
        inner = np.concatenate([-positions[:0:-1], positions])
        return np.concatenate([[-limit]*4, inner, [limit]*4])

    def _x_normalized(self, x):
        x = np.clip(np.asarray(x, dtype=float), *getattr(self, "x_range", (0.0, self.nx - 1.0)))
        return 2.0*x/max(self.nx - 1, 1) - 1.0

    def _basis(self, d):
        limit = self.half_width + 0.5
        return interpolate.BSpline.design_matrix(np.clip(d, -limit, limit), self.knots, 3).toarray()

    def _design(self, d, u):
        basis = self._basis(d)                                        # (n, n_basis)
        legendre = np.polynomial.legendre.legvander(u, self.x_order)  # (n, x_order+1)
        return (basis[:, :, None]*legendre[:, None, :]).reshape(len(d), -1)

    def profile_at(self, dy_grid, x):

        """Profile (not normalized) on a grid of dy values, at the column x (for plots)."""

        legendre = np.polynomial.legendre.legvander(np.atleast_1d(self._x_normalized(x)), self.x_order)[0]
        return np.clip(self._basis(dy_grid) @ (self.coefficients @ legendre), 0, None)

    def evaluate(self, dy, x_columns):

        """Profile at every pixel of the band (negative values set to zero; not normalized)."""

        limit = self.half_width + 0.5
        inside = np.abs(dy) <= limit
        legendre = np.polynomial.legendre.legvander(self._x_normalized(x_columns), self.x_order)   # (nx, x_order+1)
        column_coefficients = legendre @ self.coefficients.T                                      # (nx, n_basis)
        profile = np.zeros(dy.shape)
        rows_index, columns_index = np.nonzero(inside)
        basis = self._basis(dy[inside])                                                           # (n, n_basis)
        profile[rows_index, columns_index] = np.sum(basis*column_coefficients[columns_index], axis=1)
        return np.clip(profile, 0.0, None)


def horne_extraction(data, sky, variance, profile, coverage, good, sky_design, sky_covariance, reject_sigma=5.0, max_iterations=10,
                     gain_factor=None):

    """
    Optimal extraction (Horne 1986), column by column:

        F = sum(M P (D - S) / V) / sum(M P^2 / V),          Var(F) = 1 / sum(M P^2 / V)  (+ sky term)

    where D is the data, S the sky, P the spatial profile normalized to sum(coverage * P) = 1 within the aperture, V the variance and M = 1
    for the good pixels and 0 for the masked/rejected ones. If gain_factor is given, the variance is recomputed in each iteration from the
    model F P + S (see estimate_variance_model and model_variance), which removes the bias caused by noisy variances. In each
    iteration, the pixel of each column with the largest normalized residual |D - S - F P|/sqrt(V) is rejected if it exceeds reject_sigma
    (cosmic rays, hot pixels).

    The sky is a fit to other pixels, so its error is correlated between the pixels of a column; its contribution to Var(F) is computed
    exactly from the covariance of the sky coefficients.

    Returns (flux, flux_variance, used, variance) where 'used' flags the pixels kept in the extraction and 'variance' is the final
    (model) variance of each pixel.
    """

    in_aperture = coverage > 0
    used = good & in_aperture & np.isfinite(data) & np.isfinite(sky) & np.isfinite(variance) & (variance > 0)
    signal = data - sky
    given_variance = variance
    variance = model_variance(given_variance, data, sky, gain_factor) if gain_factor is not None else given_variance.copy()
    flux = np.zeros(data.shape[1])
    for iteration in range(max_iterations + 1):
        weight = np.where(used, profile/np.where(used, variance, 1.0), 0.0)
        denominator = np.sum(weight*profile, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            flux = np.sum(weight*np.where(used, signal, 0.0), axis=0)/denominator
        if gain_factor is not None:
            variance = model_variance(given_variance, data, np.nan_to_num(flux)[None, :]*profile + sky, gain_factor)
        residual = np.where(used, (signal - flux[None, :]*profile)/np.sqrt(variance), 0.0)
        worst = np.argmax(np.abs(residual), axis=0)
        columns = np.arange(data.shape[1])
        reject = np.abs(residual[worst, columns]) > reject_sigma
        if iteration == max_iterations or not np.any(reject):
            break
        used[worst[reject], columns[reject]] = False

    weight = np.where(used, profile/np.where(used, variance, 1.0), 0.0)
    denominator = np.sum(weight*profile, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.sum(weight*np.where(used, signal, 0.0), axis=0)/denominator
        w = weight/denominator[None, :]                                   # F = sum(w (D - S))
        pixel_term = np.sum(w**2*variance, axis=0)
        g = np.einsum("yx,yxk->xk", w, sky_design)                       # Sensitivity of F to the sky coefficients
        sky_term = np.einsum("xk,xkl,xl->x", g, np.nan_to_num(sky_covariance), g)
    return flux, pixel_term + sky_term, used, variance


def boxcar_extraction(data, sky, variance, profile, coverage, used, sky_design, sky_covariance):

    """
    Simple (tophat) extraction over the same aperture: F = sum(coverage (D - S)) over the good pixels, divided by the fraction of the
    profile covered by them, sum(coverage P) (which is 1 when no pixel is masked). Pixels rejected by the optimal extraction (cosmic rays)
    are also excluded. Var(F) includes the correlated sky term.
    """

    covered = np.sum(np.where(used, coverage*profile, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(used, coverage, 0.0)/covered[None, :]
        flux = np.sum(w*np.where(used, data - sky, 0.0), axis=0)
        pixel_term = np.sum(w**2*np.where(used, variance, 0.0), axis=0)
        g = np.einsum("yx,yxk->xk", w, sky_design)
        sky_term = np.einsum("xk,xkl,xl->x", g, np.nan_to_num(sky_covariance), g)
    return flux, pixel_term + sky_term


def _gaussian_with_constant(x, amplitude, center, sigma, constant):
    return amplitude*np.exp(-0.5*((x - center)/sigma)**2) + constant


def refine_peak_positions(spectrum, peaks, half_window=4):

    """
    Sub-pixel position of each peak of a (lamp) spectrum, from a fit of a Gaussian plus a constant within +-half_window pixels of the integer
    peak position. If the fit fails (or moves the peak by more than one pixel), a parabola through the three highest pixels is used instead,
    so that the number and the order of the peaks are always preserved.
    """

    x = np.arange(len(spectrum), dtype=float)
    refined = []
    for p in np.asarray(peaks, dtype=int):
        a, b = max(0, p - half_window), min(len(spectrum), p + half_window + 1)
        xx, yy = x[a:b], spectrum[a:b]
        center = float(p)
        if 0 < p < len(spectrum) - 1:
            l, c, r = spectrum[p - 1], spectrum[p], spectrum[p + 1]
            denominator = l - 2*c + r
            if denominator < 0:
                center = p + float(np.clip(0.5*(l - r)/denominator, -0.5, 0.5))
        if len(xx) >= 5:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p0 = [spectrum[p] - np.min(yy), center, 1.5, np.min(yy)]
                    popt, _ = optimize.curve_fit(_gaussian_with_constant, xx, yy, p0=p0, maxfev=2000)
                if abs(popt[1] - center) < 1.0 and 0.3 < abs(popt[2]) < half_window and popt[0] > 0:
                    center = float(popt[1])
            except (RuntimeError, ValueError):
                pass
        refined.append(center)
    return np.array(refined)
