"""
Flux calibration with spectrophotometric standard stars.

The archival spectra of the standard stars (folder "standards", from the IRAF "onedstds" library) give the AB magnitude of the star
averaged over bandpasses of a given width (typically 16-50 Angstroms). The observed spectrum of the standard star is therefore
integrated over exactly the same bandpasses, which makes the comparison independent of the spectral resolution and of the absorption
lines of the star (the line is inside the bandpass in both). For each bandpass, the sensitivity is

    S = F_lambda(archive) / C_lambda,

where C_lambda are the observed counts per second per Angstrom (after the atmospheric extinction correction). The sensitivity curve,
in magnitudes (2.5 log10 S), is fitted with a weighted cubic B-spline with robust (non-cumulative) sigma clipping. The covariance of
the spline coefficients gives the 1-sigma uncertainty of the calibration at each wavelength. The target spectra are then calibrated as

    F_lambda(target) = S(lambda) * counts / (exposure time * |d lambda/dx|),

evaluated at the wavelengths of each target spectrum (so the target and the standard star do not need the same wavelength solution).
"""

import re
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from scipy.interpolate import BSpline

FLAM_UNIT = u.erg/u.cm**2/u.s/u.AA

# Telluric absorption bands (air wavelengths, Angstrom): O2 B and A bands and the strongest H2O bands. Their depth changes with time and
# airmass, so they are excluded from the sensitivity fit by default (the curve is interpolated across them). Note that the telluric
# absorption is NOT removed from the target spectra.
TELLURIC_BANDS = [[6855.0, 6960.0], [7160.0, 7340.0], [7585.0, 7705.0], [8120.0, 8350.0], [8940.0, 9800.0]]


def _normalize_name(name):

    """Lower case, without spaces and punctuation: "BD+28 4211" -> "bd284211"."""

    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _name_variants(name):

    """Usual spellings of a standard star name in the IRAF library (e.g. LTT 4364 -> l4364, Feige 110 -> f110)."""

    normalized = _normalize_name(name)
    variants = {normalized}
    replacements = [("ltt", "l"), ("feige", "f"), ("hiltner", "hilt"), ("hiltner", "hil"), ("wolf", "w"), ("ross", "r"), ("kopff", "k")]
    for long, short in replacements:
        if normalized.startswith(long):
            variants.add(short + normalized[len(long):])
        if normalized.startswith(short) and not normalized.startswith(long):
            variants.add(long + normalized[len(short):])
    return variants


def _standard_star_name(file_name):

    """Name of the star in the first comment line of an archival file (e.g. "# LTT 4364"), or '' if there is none."""

    with open(file_name) as text_file:
        for line in text_file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") and "type" not in line and "units" not in line:
                return line.lstrip("#").strip()
            return ""
    return ""


def find_standard_files(star_name, library_path):

    """
    Search all the datasets of the library for the archival files of a standard star.

    Parameters
    ----------
    star_name: string
        Name of the star, as usually written: e.g. "LTT 4364", "Feige 110", "BD+28 4211", "EG 21", "HR 1544". Spaces, signs and case do
        not matter.
    library_path: Path
        Folder with the datasets (one subfolder per dataset).

    Returns
    -------
    matches: list
        List of (dataset, file name, star name in the file, first wavelength, last wavelength), sorted by dataset. Exact matches are given if
        there are any; otherwise, partial matches (e.g. "CD-32 9927" matches the file "cd32.dat").
    """

    library_path = Path(library_path)
    variants = _name_variants(star_name)
    exact, partial = [], []
    for dataset in sorted(p for p in library_path.iterdir() if p.is_dir()):
        aliases = {}
        names_file = dataset / "names.men"
        if names_file.exists():
            for line in names_file.read_text().splitlines():
                tokens = line.split()
                if len(tokens) == 2 and not line.startswith("#"):
                    aliases[_normalize_name(tokens[0])] = _normalize_name(tokens[1])
        for file_name in sorted(dataset.glob("*.dat")):
            stem = _normalize_name(file_name.stem)
            base = re.sub(r"(blue|red)$", "", stem)
            file_star = _standard_star_name(file_name)
            keys = {stem, base, _normalize_name(file_star)}
            keys.discard("")
            names = keys | {alias for alias, target in aliases.items() if target in keys}
            if names & variants:
                category = exact
            elif any(len(k) >= 3 and re.search(r"\d", k) and any(v.startswith(k) and len(v) > len(k) for v in variants) for k in keys):
                category = partial
            else:
                continue
            try:
                wavelength, _, _ = read_standard_file(file_name)
                limits = (float(wavelength.min()), float(wavelength.max()))
            except ValueError:
                limits = (np.nan, np.nan)
            category.append((dataset.name, file_name.name, file_star, *limits))
    return exact if exact else partial


def read_standard_file(file_name, column_unit=u.ABmag):

    """
    Read an archival spectrum of a standard star.

    Parameters
    ----------
    file_name: string or Path
        Text file with (at least) two columns: the wavelength (in Angstroms, or in microns if the file has the comment "# units microns")
        and the AB magnitude (or a flux density, see 'column_unit'). A third column, if present, is the width of each bandpass in
        Angstroms; otherwise, the spacing between the wavelengths is used.
    column_unit: astropy unit
        Unit of the second column: u.ABmag (default, as in all the files of the easyspec library) or a flux density unit, e.g.
        u.erg/u.cm**2/u.s/u.AA or u.mJy.

    Returns
    -------
    wavelength: array
        Central wavelengths of the bandpasses (Angstrom), in increasing order.
    flux: array
        Flux density F_lambda at each bandpass (erg/cm2/s/Angstrom).
    width: array
        Width of each bandpass (Angstrom).
    """

    scale = 1.0
    with open(file_name) as text_file:
        lines = text_file.read().splitlines()
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "type blackbody" in stripped.lower():
                raise ValueError(f"{Path(file_name).name} is a blackbody template (relative magnitudes in a photometric band), not the absolute "
                                 "spectrum of a star, so it cannot be used for the flux calibration. Please use the file of your standard star.")
            if re.search(r"units\s+microns", stripped.lower()):
                scale = 1e4
            continue
        rows.append([float(value) for value in stripped.split()])
    if not rows:
        raise ValueError(f"No data found in {file_name}.")
    n_columns = min(len(r) for r in rows)
    if n_columns < 2:
        raise ValueError(f"{Path(file_name).name} must have at least two columns (wavelength and magnitude/flux).")
    data = np.array([r[:min(n_columns, 3)] for r in rows], dtype=float)
    wavelength = data[:, 0]*scale
    order = np.argsort(wavelength)
    data, wavelength = data[order], wavelength[order]
    width = data[:, 2]*scale if n_columns >= 3 else np.abs(np.gradient(wavelength))
    values = data[:, 1]
    if column_unit == u.ABmag:
        flux = (values*u.ABmag).to(FLAM_UNIT, equivalencies=u.spectral_density(wavelength*u.AA)).value
    else:
        flux = (values*column_unit).to(FLAM_UNIT, equivalencies=u.spectral_density(wavelength*u.AA)).value
    good = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0) & np.isfinite(width) & (width > 0)
    return wavelength[good], flux[good], width[good]


def pixel_edges(wavelength):

    """Wavelength edges of the pixels (midpoints between the pixel centers; the first and last pixels are assumed symmetric)."""

    wavelength = np.asarray(wavelength, dtype=float)
    middle = 0.5*(wavelength[1:] + wavelength[:-1])
    return np.concatenate([[wavelength[0] - (middle[0] - wavelength[0])], middle, [wavelength[-1] + (wavelength[-1] - middle[-1])]])


def bandpass_average(wavelength, counts, variance, bad, centers, widths):

    """
    Average of a spectrum over bandpasses, accounting for the fraction of each pixel inside the bandpass.

    Parameters
    ----------
    wavelength: array
        Wavelength of each pixel (increasing or decreasing).
    counts: array
        Counts per pixel (e.g. per second).
    variance: array or None
        Variance of 'counts'.
    bad: array of bool
        Pixels that must not be used.
    centers, widths: arrays
        Bandpasses [center - width/2, center + width/2].

    Returns
    -------
    density: array
        Mean counts per unit wavelength in each bandpass (np.nan if the bandpass is not fully covered by the spectrum).
    density_variance: array
        Its variance (np.nan if 'variance' is None).
    bad_fraction: array
        Fraction of the bandpass covered by bad pixels.
    """

    edges = pixel_edges(wavelength)
    low, high = np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:])
    size = high - low
    band_low, band_high = centers - widths/2, centers + widths/2
    overlap = np.clip(np.minimum(high[None, :], band_high[:, None]) - np.maximum(low[None, :], band_low[:, None]), 0, None)
    fraction = overlap/size[None, :]                  # Fraction of each pixel inside each bandpass
    good = ~np.asarray(bad, dtype=bool) & np.isfinite(counts)
    covered = overlap.sum(axis=1)                     # Wavelength range of the bandpass covered by the spectrum
    good_covered = (overlap*good[None, :]).sum(axis=1)
    counts_good = np.where(good, counts, 0.0)
    total = fraction @ counts_good
    density = np.where((covered >= 0.999*widths) & (good_covered > 0), total/np.where(good_covered > 0, good_covered, 1), np.nan)
    if variance is not None:
        variance_good = np.where(good & np.isfinite(variance), variance, 0.0)
        density_variance = (fraction**2 @ variance_good)/np.where(good_covered > 0, good_covered, 1)**2
        density_variance = np.where(np.isfinite(density), density_variance, np.nan)
    else:
        density_variance = np.full(len(centers), np.nan)
    bad_fraction = 1 - good_covered/np.where(covered > 0, covered, 1)
    return density, density_variance, bad_fraction


class SensitivityFunction:

    """
    Sensitivity curve of the instrument: S(lambda) = F_lambda/(counts/s/Angstrom), modelled as a cubic B-spline in magnitudes,
    M(lambda) = 2.5 log10 S(lambda).

    Calling the object, sensitivity(wavelength), returns S in (erg/cm2/s/Angstrom)/(counts/s/Angstrom). The methods relative_error()
    and in_range() give the 1-sigma relative uncertainty of S and whether a wavelength is inside the calibrated range. The details of the
    fit (bandpasses, sensitivity points, lines used/rejected, residuals) are in the attribute 'info'.
    """

    def __init__(self, knots, coefficients, covariance, wavelength_range, info):
        self.knots = np.asarray(knots, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.wavelength_range = (float(wavelength_range[0]), float(wavelength_range[1]))
        self.fit_range = (float(self.knots[0]), float(self.knots[-1]))   # Range of the points used in the fit
        self.info = info
        self.unit = FLAM_UNIT/(u.ct/u.s/u.AA)
        basis = BSpline(self.knots, np.eye(len(self.coefficients)), 3, extrapolate=False)
        self._spline_basis, self._derivative_basis = basis, basis.derivative()

    def _basis(self, wavelength):
        """
        Basis functions at 'wavelength' (shape: n_wavelengths x n_coefficients). Beyond the range of the points used in the fit (e.g. when
        the last bandpasses fall in a telluric band), the curve is extrapolated linearly, and its uncertainty grows accordingly.
        """
        wavelength = np.clip(np.atleast_1d(np.asarray(wavelength, dtype=float)), *self.wavelength_range)
        low, high = self.fit_range
        basis = self._spline_basis(np.clip(wavelength, low, high))
        for edge, outside in ((low, wavelength < low), (high, wavelength > high)):
            if np.any(outside):
                basis[outside] = self._spline_basis(edge)[None, :] + (wavelength[outside] - edge)[:, None]*self._derivative_basis(edge)[None, :]
        return basis

    def magnitude(self, wavelength):
        """2.5 log10 of the sensitivity."""
        return self._basis(wavelength) @ self.coefficients

    def magnitude_error(self, wavelength):
        """1-sigma uncertainty of magnitude(), from the covariance of the spline coefficients."""
        basis = self._basis(wavelength)
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", basis, self.covariance, basis), 0, None))

    def relative_error(self, wavelength):
        """1-sigma relative uncertainty of the sensitivity (sigma_S/S)."""
        return 0.4*np.log(10)*self.magnitude_error(wavelength)

    def in_range(self, wavelength):
        """True for the wavelengths inside the range covered by the bandpasses used in the fit."""
        wavelength = np.asarray(wavelength, dtype=float)
        return (wavelength >= self.wavelength_range[0]) & (wavelength <= self.wavelength_range[1])

    def __call__(self, wavelength):
        if isinstance(wavelength, u.Quantity):
            wavelength = wavelength.to(u.AA).value
        return 10**(0.4*self.magnitude(wavelength))*self.unit


def _knots(x, spacing, min_points):

    """Interior knots every 'spacing' Angstroms, merging the intervals with fewer than 'min_points' points (e.g. across gaps)."""

    knots = list(np.arange(x.min() + spacing, x.max() - 0.5*spacing, spacing))
    changed = True
    while changed and knots:
        changed = False
        boundaries = np.concatenate([[x.min() - 1], knots, [x.max() + 1]])
        counts = np.histogram(x, bins=boundaries)[0]
        worst = int(np.argmin(counts))
        if counts[worst] < min_points:
            # Remove the knot that closes the emptiest interval (the right one, except for the last interval)
            del knots[min(worst, len(knots) - 1)]
            changed = True
    return np.array(knots)


def fit_sensitivity(wavelength, magnitude, magnitude_error, candidates, knot_spacing, clip_sigma=3.5, max_iterations=10):

    """
    Weighted least-squares fit of a cubic B-spline to the sensitivity points (in magnitudes), with robust sigma clipping.

    Parameters
    ----------
    wavelength, magnitude, magnitude_error: arrays
        The sensitivity points (one per bandpass) and their 1-sigma uncertainties.
    candidates: array of bool
        Points that can be used (e.g. not in excluded regions).
    knot_spacing: float
        Approximate distance between the knots of the spline (Angstrom).
    clip_sigma: float
        Points deviating more than clip_sigma (in units of their uncertainty, scaled by the scatter of the residuals) are rejected. The
        rejection is recomputed from all the candidate points at each iteration (non-cumulative), so a point rejected because of a bad
        intermediate fit can come back.

    Returns
    -------
    knots, coefficients, covariance, used, reduced_chi2
    """

    x, y, sigma = np.asarray(wavelength), np.asarray(magnitude), np.asarray(magnitude_error)
    candidates = candidates & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if np.count_nonzero(candidates) < 6:
        raise RuntimeError(f"Only {np.count_nonzero(candidates)} bandpasses of the archival spectrum can be used (at least 6 are needed). Check "
                           "that the wavelength range of the standard star spectrum overlaps with the archival file, and the excluded regions.")
    def fit(used):
        spacing = knot_spacing
        while True:
            interior = _knots(x[used], spacing, min_points=3)
            if np.count_nonzero(used) >= len(interior) + 4 + 2:
                break
            spacing *= 1.5
        knots = np.concatenate([[x[candidates].min()]*4, interior, [x[candidates].max()]*4])
        n_coefficients = len(knots) - 4
        basis_all = BSpline(knots, np.eye(n_coefficients), 3, extrapolate=True)(np.clip(x, knots[0], knots[-1]))
        weights = 1/sigma[used]
        design = basis_all[used]*weights[:, None]
        coefficients = np.linalg.lstsq(design, y[used]*weights, rcond=None)[0]
        normalized = (y - basis_all @ coefficients)/sigma
        reduced_chi2 = float(np.sum(normalized[used]**2)/max(1, np.count_nonzero(used) - n_coefficients))
        # The scatter of the residuals beyond the uncertainties (e.g. errors of the archival fluxes) is included in the covariance:
        covariance = np.linalg.pinv(design.T @ design)*max(1.0, reduced_chi2)
        return knots, coefficients, covariance, normalized, reduced_chi2

    used = candidates.copy()
    for _ in range(max_iterations):
        knots, coefficients, covariance, normalized, reduced_chi2 = fit(used)
        # Robust scatter of the normalized residuals (the uncertainties are never assumed to be overestimated):
        scale = max(1.0, 1.4826*np.median(np.abs(normalized[used])))
        new_used = candidates & (np.abs(normalized) <= clip_sigma*scale)
        if np.array_equal(new_used, used) or np.count_nonzero(new_used) < 6:
            break
        used = new_used
    knots, coefficients, covariance, normalized, reduced_chi2 = fit(used)
    return knots, coefficients, covariance, used, reduced_chi2


def regions_mask(wavelength, regions):

    """True for the wavelengths inside any of the regions [[w1, w2], ...] (a single region [w1, w2] is also accepted)."""

    wavelength = np.asarray(wavelength, dtype=float)
    mask = np.zeros(wavelength.shape, dtype=bool)
    if regions is None or len(regions) == 0:
        return mask
    if np.ndim(regions[0]) == 0:
        regions = [regions]
    for region in regions:
        if len(region) != 2:
            raise ValueError("Each region must be given as [lambda_min, lambda_max], e.g. [[6860, 6960], [7590, 7700]].")
        mask |= (wavelength >= min(region)) & (wavelength <= max(region))
    return mask


def bands_overlap(centers, widths, regions):

    """True for the bandpasses that overlap any of the regions."""

    mask = np.zeros(len(centers), dtype=bool)
    if regions is None or len(regions) == 0:
        return mask
    if np.ndim(regions[0]) == 0:
        regions = [regions]
    for region in regions:
        mask |= (centers + widths/2 > min(region)) & (centers - widths/2 < max(region))
    return mask


def compute_sensitivity(wavelength, counts_per_second, variance_per_second2, bad, archive_wavelength, archive_flux, archive_width,
                        exclude_regions=None, mask_telluric=True, knot_spacing=None, clip_sigma=3.5, archive_uncertainty=0.01,
                        max_bad_fraction=0.1):

    """
    Sensitivity function from the observed spectrum of a standard star (see the module docstring).

    Parameters
    ----------
    wavelength: array
        Wavelength of each pixel of the standard star spectrum (Angstrom).
    counts_per_second: array
        Spectrum of the standard star (counts/s per pixel), corrected for the atmospheric extinction.
    variance_per_second2: array or None
        Its variance.
    bad: array of bool
        Pixels that must not be used (e.g. from the extraction mask).
    archive_wavelength, archive_flux, archive_width: arrays
        Archival spectrum (output of read_standard_file()).
    exclude_regions: list
        Regions (Angstrom) to exclude, e.g. [[6860, 6960]]. Bandpasses overlapping them are not used.
    mask_telluric: bool
        If True, the bandpasses overlapping the telluric bands (TELLURIC_BANDS) are also excluded.
    knot_spacing: float
        Distance between the knots of the spline (Angstrom). Default: max(4 x median spacing of the bandpasses, 1/20 of the range, 150 A).
    clip_sigma: float
        Threshold of the sigma clipping.
    archive_uncertainty: float
        Relative uncertainty of the archival fluxes (added in quadrature to the statistical uncertainty of each point). Default is 1%.
    max_bad_fraction: float
        Bandpasses with a larger fraction covered by bad pixels are not used.

    Returns
    -------
    sensitivity: SensitivityFunction
    """

    wavelength = np.asarray(wavelength, dtype=float)
    dispersion = np.abs(np.gradient(wavelength))
    if np.any(dispersion <= 0) or not np.all(np.isfinite(wavelength)):
        raise ValueError("The wavelength solution of the standard star must be finite and strictly monotonic.")

    # Effective width: bandpasses narrower than 2 pixels (e.g. the 1 A "bandpasses" of oke1990) are widened to 2 pixels.
    local_dispersion = np.interp(archive_wavelength, np.sort(wavelength), dispersion[np.argsort(wavelength)])
    widths = np.maximum(archive_width, 2*local_dispersion)
    density, density_variance, bad_fraction = bandpass_average(wavelength, counts_per_second, variance_per_second2, bad,
                                                               archive_wavelength, widths)

    observed = np.isfinite(density) & (density > 0) & (bad_fraction <= max_bad_fraction)
    excluded = bands_overlap(archive_wavelength, widths, exclude_regions)
    telluric = bands_overlap(archive_wavelength, widths, TELLURIC_BANDS) if mask_telluric else np.zeros(len(widths), dtype=bool)
    candidates = observed & ~excluded & ~telluric
    if np.count_nonzero(observed) == 0:
        raise RuntimeError(f"The standard star spectrum ({wavelength.min():.0f}-{wavelength.max():.0f} A) does not overlap with any bandpass of "
                           f"the archival spectrum ({archive_wavelength.min():.0f}-{archive_wavelength.max():.0f} A), or its counts are not "
                           "positive. Check the archival file and the wavelength solution of the standard star.")

    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = 2.5*np.log10(archive_flux/density)
        relative_statistical = np.sqrt(density_variance)/density
    relative_statistical = np.where(np.isfinite(relative_statistical), relative_statistical, 0.0)
    magnitude_error = 2.5/np.log(10)*np.sqrt(relative_statistical**2 + archive_uncertainty**2)
    if not np.any(np.isfinite(density_variance[observed])):
        warnings.warn("The variance of the standard star spectrum is unknown: only 'archive_uncertainty' is used as the uncertainty of each "
                      "sensitivity point.")

    if knot_spacing is None:
        spacing = np.median(np.diff(archive_wavelength[candidates])) if np.count_nonzero(candidates) > 1 else 50.0
        span = np.ptp(archive_wavelength[candidates]) if np.count_nonzero(candidates) > 1 else 1000.0
        knot_spacing = max(4*spacing, span/20, 150.0)

    knots, coefficients, covariance, used, reduced_chi2 = fit_sensitivity(archive_wavelength, magnitude, magnitude_error, candidates,
                                                                          knot_spacing, clip_sigma=clip_sigma)
    # Calibrated range: the bandpasses observed with good data, including those excluded from the fit at the edges (e.g. a telluric band
    # at the red end), where the curve is extrapolated linearly (see SensitivityFunction._basis()).
    usable = observed & ~excluded   # (telluric bands are allowed, user-excluded regions are not)
    first = np.argmin(np.where(usable, archive_wavelength, np.inf))
    last = np.argmax(np.where(usable, archive_wavelength, -np.inf))
    wavelength_range = (archive_wavelength[first] - widths[first]/2, archive_wavelength[last] + widths[last]/2)
    # The calibrated range cannot go beyond the standard star spectrum:
    wavelength_range = (max(wavelength_range[0], wavelength.min()), min(wavelength_range[1], wavelength.max()))
    info = {"band_wavelength": archive_wavelength, "band_width": widths, "archive_flux": archive_flux, "observed_density": density,
            "observed_density_variance": density_variance, "magnitude": magnitude, "magnitude_error": magnitude_error,
            "candidates": candidates, "used": used, "excluded": excluded | telluric, "observed": observed,
            "extrapolated_range": None,
            "reduced_chi2": reduced_chi2, "knot_spacing": knot_spacing, "n_knots": len(knots) - 8}
    sensitivity = SensitivityFunction(knots, coefficients, covariance, wavelength_range, info)
    extrapolated = []
    if sensitivity.fit_range[0] - wavelength_range[0] > widths[first]:
        extrapolated.append((wavelength_range[0], sensitivity.fit_range[0]))
    if wavelength_range[1] - sensitivity.fit_range[1] > widths[last]:
        extrapolated.append((sensitivity.fit_range[1], wavelength_range[1]))
    info["extrapolated_range"] = extrapolated
    residuals = magnitude - sensitivity.magnitude(archive_wavelength)
    info["residuals"] = residuals
    info["rms_percent"] = float(100*0.4*np.log(10)*np.sqrt(np.mean(residuals[used]**2)))
    return sensitivity
