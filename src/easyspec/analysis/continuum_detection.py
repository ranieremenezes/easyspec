"""
Continuum estimation and line detection for easyspec.analysis.

The main tools are:

- find_continuum_mask(): automatic identification of the continuum (line-free) pixels of a spectrum, for any kind of object (AGN with broad
  emission lines, stars with absorption lines, galaxies with narrow lines). A smooth curve (a least-squares cubic spline with widely spaced
  knots, which cannot follow the lines) is fitted iteratively to the spectrum. In each iteration, the residuals are smoothed on several scales
  (from a few pixels up to a fraction of the knot spacing), each scale normalized by the robust scatter of the smoothed residuals in the
  continuum, and the lines are detected with hysteresis thresholding: a line is seeded where the
  smoothed residual exceeds 'detection_sigma' and extended, with the same sign, while it exceeds 'extension_sigma'. This captures both narrow
  lines and the faint wings of broad lines. The masked pixels are excluded from the next fit, until the mask converges.
- fit_continuum(): the continuum fitted to the continuum pixels with a polynomial, a smoothing spline or a median filter.
- detect_lines(): emission and absorption lines detected on the full continuum-subtracted spectrum (never on separate segments), with the
  noise of the continuum interpolated at each wavelength.
"""


import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import BSpline, make_lsq_spline


def find_nearest(array, value):

    """Index of the element of 'array' closest to 'value' (NaNs are ignored)."""

    return int(np.nanargmin(np.abs(np.asarray(array, dtype=float) - value)))


def regions_to_mask(wavelengths, regions):

    """Boolean mask of the wavelengths inside any of the regions [[w1, w2], ...] (a single region [w1, w2] is also accepted)."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    mask = np.zeros(wavelengths.shape, dtype=bool)
    if regions is None or len(regions) == 0:
        return mask
    if np.ndim(regions[0]) == 0:
        regions = [regions]
    for region in regions:
        if len(region) != 2:
            raise ValueError("Each continuum region must be given as [lambda_min, lambda_max], e.g. [[3000, 6830], [6930, 7500]].")
        mask |= (wavelengths >= min(region)) & (wavelengths <= max(region))
    return mask


def mask_to_regions(wavelengths, mask, min_pixels=5):

    """Contiguous runs of True in 'mask' (with at least min_pixels pixels), as a list of [lambda_min, lambda_max] in increasing order."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    labels, n_runs = ndimage.label(mask)
    regions = []
    for run in range(1, n_runs + 1):
        index = np.nonzero(labels == run)[0]
        if len(index) >= min_pixels:
            regions.append([float(wavelengths[index].min()), float(wavelengths[index].max())])
    return sorted(regions)


def empirical_noise(flux, good, window=51):

    """
    Noise of each pixel estimated from the data themselves: robust scatter (MAD) of the differences between neighbouring pixels, in a running
    window. The differences remove any smooth signal (continuum and resolved lines).
    """

    flux = np.asarray(flux, dtype=float)
    differences = np.full(len(flux), np.nan)
    pair_good = good[1:] & good[:-1]
    differences[1:][pair_good] = np.abs(np.diff(flux))[pair_good]
    window = int(max(5, min(window, len(flux)//3)))
    local = ndimage.generic_filter(differences, lambda d: np.nanmedian(d) if np.any(np.isfinite(d)) else np.nan, size=window, mode="constant", cval=np.nan)
    noise = 1.4826*local/np.sqrt(2)
    if not np.any(np.isfinite(noise)):
        return np.full(len(flux), np.nanstd(flux[good]))
    fill = np.nanmedian(noise)
    return np.where(np.isfinite(noise) & (noise > 0), noise, fill)


def _spline_fit(x, y, weights, use, knot_spacing):

    """Weighted least-squares cubic spline with interior knots every 'knot_spacing' (only knots with enough points around them are kept)."""

    xs = x[use]
    interior = np.arange(xs.min() + knot_spacing, xs.max() - 0.5*knot_spacing, knot_spacing)
    # Schoenberg-Whitney conditions: each interval between knots must contain points
    if len(interior):
        counts = np.histogram(xs, bins=np.concatenate([[xs.min()], interior, [xs.max()]]))[0]
        while len(interior) and np.any(counts < 4):
            interior = np.delete(interior, min(int(np.argmin(counts)), len(interior) - 1))
            counts = np.histogram(xs, bins=np.concatenate([[xs.min()], interior, [xs.max()]]))[0]
    knots = np.concatenate([[xs.min()]*4, interior, [xs.max()]*4])
    spline = make_lsq_spline(xs, y[use], knots, k=3, w=weights[use])
    curve = BSpline(spline.t, spline.c, 3, extrapolate=False)
    slope = curve.derivative()
    values = curve(np.clip(x, xs.min(), xs.max()))
    # Linear extrapolation beyond the points used (a cubic extrapolation diverges quickly):
    below, above = x < xs.min(), x > xs.max()
    values[below] += slope(xs.min())*(x[below] - xs.min())
    values[above] += slope(xs.max())*(x[above] - xs.max())
    return values


def _hysteresis(significance, detection_sigma, extension_sigma):

    """Pixels connected (with the same sign) to a seed above detection_sigma, while staying above extension_sigma."""

    mask = np.zeros(len(significance), dtype=bool)
    for sign in (1, -1):
        candidate = sign*significance > extension_sigma
        labels, n_runs = ndimage.label(candidate)
        if n_runs == 0:
            continue
        seeds = np.unique(labels[(sign*significance > detection_sigma) & candidate])
        mask |= np.isin(labels, seeds[seeds > 0])
    return mask


def find_continuum_mask(wavelengths, flux, flux_err=None, good=None, knot_spacing=None, detection_sigma=3.0, extension_sigma=1.0,
                        scales=None, padding=2, max_iterations=15, min_continuum_fraction=0.25, max_line_velocity=30000.0):

    """
    Automatic identification of the continuum (line-free) pixels of a spectrum (see the module docstring).

    Parameters
    ----------
    wavelengths, flux: arrays
        The spectrum.
    flux_err: array
        Optional. 1-sigma uncertainty of each pixel. If None (or not finite), the noise is estimated from the data.
    good: array of bool
        Optional. Pixels that can be used (e.g. from the data quality mask).
    knot_spacing: float
        Distance between the knots of the smooth curve (same units as the wavelengths). It must be larger than the full width of the broadest
        lines (including their wings). Default: 1/6 of the wavelength range (at least 40 pixels).
    detection_sigma, extension_sigma: floats
        Thresholds (in units of the noise of the smoothed residuals) to seed and to extend a line.
    scales: list
        Smoothing scales (in pixels). Default: 1, 2, 4, ... up to ~1/4 of the knot spacing.
    padding: int
        Pixels added on both sides of each detected line.
    max_line_velocity: float
        Maximum full width (in km/s) of a line. Broader "lines" are taken as a mismatch of the smooth curve and are not masked. Default is
        30000 km/s (enough for the broad lines of AGN, including their wings).
    min_continuum_fraction: float
        The iterations stop before the continuum pixels become fewer than this fraction of the spectrum (for spectra dominated by lines or
        molecular bands, where the result is a pseudo-continuum).

    Returns
    -------
    continuum_mask: array of bool
        True for the continuum pixels.
    smooth_continuum: array
        The smooth curve of the last iteration.
    noise: array
        Noise of each pixel (flux_err rescaled by the scatter of the continuum residuals, or estimated from the data).
    """

    x, f = np.asarray(wavelengths, dtype=float), np.asarray(flux, dtype=float)
    n = len(x)
    good = np.isfinite(x) & np.isfinite(f) if good is None else np.asarray(good, dtype=bool) & np.isfinite(x) & np.isfinite(f)
    if np.count_nonzero(good) < 20:
        raise ValueError("The spectrum has fewer than 20 valid pixels.")
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        valid_err = np.isfinite(flux_err) & (flux_err > 0)
        if np.count_nonzero(valid_err & good) < 0.5*np.count_nonzero(good):
            flux_err = None
    noise = empirical_noise(f, good) if flux_err is None else np.where(valid_err, flux_err, np.nanmedian(flux_err[valid_err]))
    pixel = np.nanmedian(np.abs(np.diff(x)))
    span = np.ptp(x[good])
    if knot_spacing is None:
        knot_spacing = max(span/6, 40*pixel)
    if scales is None:
        scales = [s for s in (1, 2, 4, 8, 16, 32, 64) if s*pixel <= 0.25*knot_spacing] or [1]

    line_mask = np.zeros(n, dtype=bool)
    n_good = np.count_nonzero(good)
    # Robust spline, by iteratively reweighted least squares: first with Huber weights (a convex problem, which converges from any first guess
    # and is only weakly affected by the lines), then with Tukey's biweight, which gives zero weight to the lines. The spline follows the shape
    # of the continuum over the whole spectrum (no extrapolation, even if the continuum is strongly curved).
    smooth = _spline_fit(x, f, 1/noise, good, knot_spacing)
    for iteration in range(25):
        residual = (f - smooth)/noise
        scale = 1.4826*np.median(np.abs(residual[good] - np.median(residual[good])))
        u_values = np.abs(residual)/max(scale, 1e-12)
        if iteration < 15:
            robust_weight = np.minimum(1.0, 1.345/np.maximum(u_values, 1e-12))   # Huber
        else:
            robust_weight = np.where(u_values < 4.685, (1 - (u_values/4.685)**2)**2, 0.0)   # Tukey's biweight
        use = good & (robust_weight > 0)
        if np.count_nonzero(use) < 20:
            break
        smooth = _spline_fit(x, f, np.sqrt(robust_weight)/noise, use, knot_spacing)
    def line_pixels(residual, use, detection, extension):
        """Multi-scale hysteresis detection of the lines in the normalized residuals."""
        mask = np.zeros(n, dtype=bool)
        for s in sorted(scales):
            # The pixels already masked at smaller scales are set to zero before smoothing, so that a strong narrow line is not smeared
            # into a broad mask at the larger scales (which only look for faint, broad features such as the wings of broad lines).
            remaining = np.where(mask, 0.0, residual)
            smoothed = ndimage.gaussian_filter1d(remaining, s, mode="nearest") if s > 0.5 else remaining
            # Significance at this scale: relative to the actual scatter of the smoothed residuals in the continuum (which includes any
            # correlated structure, e.g. weak features or calibration ripples), and never below the expectation for white noise.
            centre = np.median(smoothed[use])
            scatter = 1.4826*np.median(np.abs(smoothed[use] - centre))
            white = 1/np.sqrt(2*np.sqrt(np.pi)*s) if s > 0.5 else 1.0
            significance = (smoothed - centre)/max(scatter, white)
            mask_scale = _hysteresis(significance, detection, extension)
            # A "line" much broader than any real line is (mostly) a mismatch between the smooth curve and the continuum: only its core
            # (above the detection threshold) is masked, so that the next fit can follow the continuum while the line itself stays masked.
            labels, n_runs = ndimage.label(mask_scale)
            for run in range(1, n_runs + 1):
                members = np.nonzero(labels == run)[0]
                limit = max_line_velocity/299792.458*np.median(x[members])
                if members[0] == 0 or members[-1] == n - 1:
                    limit *= 0.5   # A line at the edge of the spectrum is only partially observed
                if np.ptp(x[members]) + pixel > limit:
                    mask_scale[members] = np.abs(significance[members]) > detection
            mask |= mask_scale
        if padding > 0:
            mask = ndimage.binary_dilation(mask, iterations=int(padding))
        return mask & good

    # Spectra with a lot of real structure at high S/N (e.g. late-type stars, or quasars with many weak features) could be masked almost
    # entirely: in this case the thresholds are raised (by factors of 1.5) until at least min_continuum_fraction of the pixels remain, so
    # that only the strongest features are masked.
    threshold_factor = 1.0
    for iteration in range(max_iterations):
        use = good & ~line_mask
        if iteration > 0:
            smooth = _spline_fit(x, f, 1/noise, use, knot_spacing)
        residual = np.where(good, (f - smooth)/noise, 0.0)
        for _ in range(8):
            new_mask = line_pixels(residual, use, threshold_factor*detection_sigma, threshold_factor*extension_sigma)
            if np.count_nonzero(good & ~new_mask) >= min_continuum_fraction*n_good:
                break
            threshold_factor *= 1.5
        if np.array_equal(new_mask, line_mask):
            break
        line_mask = new_mask
    continuum_mask = good & ~line_mask
    final_scale = 1.4826*np.median(np.abs(((f - smooth)/noise)[continuum_mask])) if np.any(continuum_mask) else 1.0
    return continuum_mask, smooth, noise*max(final_scale, 1e-12)


def fit_continuum(wavelengths, flux, continuum_mask, method="median_filter", pl_order=2, smooth_window=111, knot_spacing=None,
                  weights=None, clip_sigma=3.0):

    """
    Fit the continuum to the pixels in continuum_mask.

    Methods
    -------
    "polynomial" (aliases "powerlaw", "pl", "PL", "power-law"; the name "powerlaw" is kept for compatibility, but it is a polynomial of order
    pl_order in wavelength): weighted, robust (sigma-clipped) polynomial fit, in a normalized coordinate (well conditioned for any order).
    "spline": weighted least-squares cubic spline with knots every 'knot_spacing'.
    "median_filter": the continuum pixels are linearly interpolated across the lines and smoothed with a median filter of 'smooth_window' pixels.
    """

    x, f = np.asarray(wavelengths, dtype=float), np.asarray(flux, dtype=float)
    use = np.asarray(continuum_mask, dtype=bool) & np.isfinite(f) & np.isfinite(x)
    if np.count_nonzero(use) < 5:
        raise ValueError("Fewer than 5 continuum pixels: check the continuum regions.")
    w = np.ones(len(x)) if weights is None else np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    method_name = str(method).lower()
    if method_name in ("polynomial", "powerlaw", "pl", "power-law", "poly"):
        current = use.copy()
        for _ in range(10):
            polynomial = np.polynomial.Polynomial.fit(x[current], f[current], pl_order, w=w[current])
            residual = (f - polynomial(x))*w if weights is not None else f - polynomial(x)
            sigma = 1.4826*np.median(np.abs(residual[current]))
            new = use & (np.abs(residual) <= clip_sigma*max(sigma, 1e-300))
            if np.array_equal(new, current) or np.count_nonzero(new) < pl_order + 3:
                break
            current = new
        return polynomial(x)
    if method_name == "spline":
        spacing = knot_spacing if knot_spacing is not None else max(np.ptp(x[use])/8, 30*np.nanmedian(np.abs(np.diff(x))))
        return _spline_fit(x, f, w if weights is not None else np.ones(len(x)), use, spacing)
    if method_name in ("median_filter", "median"):
        order = np.argsort(x[use])
        interpolated = np.interp(x, x[use][order], f[use][order])
        window = int(smooth_window)
        if window % 2 == 0:
            window -= 1
        window = max(3, min(window, 2*(len(x)//2) - 1))
        return signal.medfilt(interpolated, window)
    raise ValueError(f"Unknown continuum method '{method}'. Options are 'median_filter', 'polynomial' (or 'powerlaw') and 'spline'.")


def region_noise(wavelengths, residual, continuum_mask, regions):

    """
    Robust noise (MAD) of the continuum residuals in each region, and the noise at each wavelength, interpolated linearly between the centres
    of the regions (constant beyond the first and last ones).
    """

    x = np.asarray(wavelengths, dtype=float)
    values, centres = [], []
    for region in regions:
        inside = continuum_mask & (x >= region[0]) & (x <= region[1]) & np.isfinite(residual)
        if np.count_nonzero(inside) >= 3:
            values.append(1.4826*np.median(np.abs(residual[inside] - np.median(residual[inside]))))
        else:
            values.append(np.nan)
        centres.append(0.5*(region[0] + region[1]))
    values, centres = np.array(values), np.array(centres)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        fallback = 1.4826*np.median(np.abs(residual[continuum_mask])) if np.any(continuum_mask) else np.nanstd(residual)
        return values, np.full(len(x), fallback)
    return values, np.interp(x, centres[valid], values[valid])


def local_noise(wavelengths, residual, continuum_mask, window=101):

    """
    Noise of the continuum at each pixel: robust scatter (MAD) of the continuum residuals within a running window of 'window' pixels,
    interpolated across the lines (where there are no continuum pixels).
    """

    residual = np.asarray(residual, dtype=float)
    values = np.where(continuum_mask & np.isfinite(residual), residual, np.nan)
    window = int(max(11, min(window, len(values))))

    def mad(chunk):
        chunk = chunk[np.isfinite(chunk)]
        return 1.4826*np.median(np.abs(chunk - np.median(chunk))) if len(chunk) >= 7 else np.nan

    # Outside the spectrum the window is padded with NaNs (ignored): padding with copies of the edge pixel ("nearest") would make the MAD
    # of the edge windows artificially close to zero.
    noise = ndimage.generic_filter(values, mad, size=window, mode="constant", cval=np.nan)
    valid = np.isfinite(noise) & (noise > 0)
    if not np.any(valid):
        return np.full(len(values), np.nanstd(residual))
    index = np.arange(len(values))
    return np.interp(index, index[valid], noise[valid])


def detect_lines(wavelengths, residual, noise, threshold_sigma=5.0, peak_distance=30, peak_width=10, good=None):

    """
    Emission and absorption lines in the continuum-subtracted spectrum 'residual', with a threshold of threshold_sigma times the local noise.
    The detection is done on the whole spectrum at once, so a line is always seen rising and falling, wherever it is.

    Returns
    -------
    index: array of int
        Pixel of each line (sorted by wavelength).
    kind: array of str
        "emission" or "absorption".
    """

    residual = np.asarray(residual, dtype=float)
    if good is not None:
        bad = ~np.asarray(good, dtype=bool) | ~np.isfinite(residual)
        if np.any(bad) and np.any(~bad):
            index = np.arange(len(residual))
            residual = residual.copy()
            residual[bad] = np.interp(index[bad], index[~bad], residual[~bad])   # Bad pixels are interpolated (no spurious peaks)
    threshold = threshold_sigma*np.asarray(noise, dtype=float)
    peaks, kinds = [], []
    for sign, kind in ((1, "emission"), (-1, "absorption")):
        # The prominence criterion (a peak must stand out by threshold_sigma from the dips that separate it from higher peaks) avoids that
        # the noise on top of a strong (e.g. broad) line produces several detections of the same line.
        found, _ = signal.find_peaks(sign*residual, height=threshold, prominence=threshold, distance=max(1, peak_distance), width=peak_width)
        peaks.extend(found.tolist())
        kinds.extend([kind]*len(found))
    if not peaks:
        return np.array([], dtype=int), np.array([], dtype=str)
    order = np.argsort(np.asarray(wavelengths, dtype=float)[peaks])
    return np.array(peaks, dtype=int)[order], np.array(kinds)[order]