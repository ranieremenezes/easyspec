"""
Auxiliary functions of easyspec.photometry for PSF ("profile") photometry: construction of an empirical PSF (photutils EPSFBuilder) from
bright isolated stars, simultaneous PSF fitting of grouped stars (photutils PSFPhotometry) with iterative detection of the stars hidden in
the residuals, and the correction from PSF fluxes to total fluxes.

Compatible with photutils 2.x and 3.x (some keyword names changed between them; they are filtered here).
"""

import inspect
import warnings

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.psf import EPSFBuilder, PSFPhotometry, SourceGrouper, extract_stars

from . import aux_photometry as aux


def _accepted_keywords(callable_object, keywords):
    """Keep only the keywords accepted by callable_object (the photutils API changed between versions 2 and 3)."""
    parameters = inspect.signature(callable_object).parameters
    return {name: value for name, value in keywords.items() if name in parameters}


def build_empirical_psf(image, sources, number_of_stars=60, stamp_size_in_fwhm=7.0, oversampling=2, maximum_iterations=10):

    """
    Empirical PSF (ePSF, Anderson & King 2000) from bright, unsaturated and isolated stars of the image (photutils EPSFBuilder).

    The stars are selected in steps of decreasing isolation (no neighbour brighter than 1%, then 5%, of the star within 5, then 3 FWHM)
    until enough are found. The background map of the image (image.background, which follows the diffuse light of clusters and galaxies)
    is subtracted before the stars are extracted. Returns the ePSF model (a fittable photutils model with parameters flux, x_0, y_0), the
    table of stars used and the stamp size.
    """

    fwhm = image.fwhm
    stamp_size = int(2*np.ceil(0.5*stamp_size_in_fwhm*fwhm) + 1)
    half_size = stamp_size//2
    selected = []
    for isolation_in_fwhm, neighbour_ratio in ((5.0, 0.005), (5.0, 0.01), (4.0, 0.02), (3.0, 0.05)):
        selected = aux.select_isolated_stars(image, sources, isolation_in_fwhm*fwhm, maximum_neighbour_ratio=neighbour_ratio,
                                             margin=half_size + 2, minimum_peak_in_sigmas=20.0)
        if len(selected) >= min(number_of_stars, 20):
            break
    selected = selected[:number_of_stars]
    if len(selected) < 5:
        raise RuntimeError(f"{image.name}: only {len(selected)} bright isolated stars to build the PSF.")
    star_table = Table()
    star_table["x"] = np.asarray(sources["x"], dtype=float)[selected]
    star_table["y"] = np.asarray(sources["y"], dtype=float)[selected]
    residual = np.where(image.bad_pixels, 0.0, image.data - image.background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stars = extract_stars(NDData(residual), star_table, size=stamp_size)
        builder = EPSFBuilder(**_accepted_keywords(EPSFBuilder, dict(oversampling=oversampling, maxiters=maximum_iterations,
                                                                      progress_bar=False)))
        empirical_psf, _ = builder(stars)
    return empirical_psf, star_table, stamp_size


def rebuild_empirical_psf(image, psf_star_table, stamp_size, residual_image, star_positions, star_fluxes, psf_model, oversampling=2,
                          maximum_iterations=10):

    """
    Rebuild the empirical PSF from its stars with all their neighbours subtracted (as DAOPHOT does): for each PSF star, the stamp is cut
    from the residual image (image - background - model of all fitted stars) and the model of the star itself is added back. In crowded
    fields this removes the light of the neighbours from the wings of the PSF, which otherwise biases all PSF fluxes.
    residual_image = image - background - model of all stars; star_positions, star_fluxes: fitted values of all stars.
    """

    tree = aux.cKDTree(star_positions)
    half_size = stamp_size//2 + 2
    cutouts, catalogs = [], []
    single = psf_model.copy()
    for x, y in zip(psf_star_table["x"], psf_star_table["y"]):
        distance, index = tree.query([x, y])
        x_center, y_center = int(round(x)), int(round(y))
        window = (slice(y_center - half_size, y_center + half_size + 1), slice(x_center - half_size, x_center + half_size + 1))
        cutout = np.array(residual_image[window], dtype=float)
        if cutout.shape != (2*half_size + 1, 2*half_size + 1) or distance > 1.0:
            continue
        y_grid, x_grid = np.mgrid[window]
        single.x_0, single.y_0, single.flux = star_positions[index, 0], star_positions[index, 1], star_fluxes[index]
        cutout += single(x_grid, y_grid)
        catalog = Table()
        catalog["x"], catalog["y"] = [star_positions[index, 0] - (x_center - half_size)], [star_positions[index, 1] - (y_center - half_size)]
        cutouts.append(NDData(np.where(np.isfinite(cutout), cutout, 0.0)))
        catalogs.append(catalog)
    if len(cutouts) < 5:
        return psf_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stars = extract_stars(cutouts, catalogs, size=stamp_size)
        builder = EPSFBuilder(**_accepted_keywords(EPSFBuilder, dict(oversampling=oversampling, maxiters=maximum_iterations,
                                                                      progress_bar=False)))
        new_psf, _ = builder(stars)
    return new_psf


def render_stars(psf_model, positions, fluxes, shape, stamp_size):
    """Image with the PSF model of each star (position, flux) added: used for residual images and neighbour subtraction."""
    image = np.zeros(shape)
    half_size = stamp_size//2
    number_of_rows, number_of_columns = shape
    model = psf_model.copy()
    for (x, y), flux in zip(positions, fluxes):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(flux)):
            continue
        x_center, y_center = int(round(x)), int(round(y))
        x_start, x_end = max(x_center - half_size, 0), min(x_center + half_size + 1, number_of_columns)
        y_start, y_end = max(y_center - half_size, 0), min(y_center + half_size + 1, number_of_rows)
        if x_end <= x_start or y_end <= y_start:
            continue
        y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
        model.x_0, model.y_0, model.flux = x, y, flux
        image[y_start:y_end, x_start:x_end] += model(x_grid, y_grid)
    return image


def adaptive_grouping_separation(positions, requested_separation, fwhm, maximum_group_size=25):
    """
    Largest grouping separation (<= requested_separation, >= 1 FWHM) for which no group of simultaneously fitted stars exceeds
    maximum_group_size. In dense cores the groups percolate (hundreds of stars linked in chains), which makes the simultaneous fit very slow
    and badly conditioned; neighbours beyond the grouping distance are still accounted for, because they are subtracted in the next
    iteration.
    """
    positions = positions[np.all(np.isfinite(positions), axis=1)]
    separation = requested_separation
    while separation > 1.0*fwhm:
        groups = SourceGrouper(min_separation=separation)(positions[:, 0], positions[:, 1])
        if np.max(np.unique(groups, return_counts=True)[1]) <= maximum_group_size:
            return separation
        separation -= 0.25*fwhm
    return 1.0*fwhm


def fit_psf_photometry(image, psf_model, initial_positions, initial_fluxes, fit_shape, error_image, grouping_separation, aperture_radius):
    """One PSFPhotometry call (simultaneous fit of grouped stars) with the given initial positions and fluxes; returns the photutils table."""
    initial = Table()
    initial["x"], initial["y"], initial["flux"] = initial_positions[:, 0], initial_positions[:, 1], initial_fluxes
    # Saturated (non-linear) pixels are excluded from the fit: saturated stars are fitted with their unsaturated wings (as in DAOPHOT)
    unusable = image.bad_pixels | (image.data > image.saturation_level)
    data = np.where(unusable, 0.0, image.data - image.background)
    # xy_bounds: the fitted position may not move more than ~1 FWHM from its initial value (prevents faint stars from running away)
    keywords = dict(grouper=SourceGrouper(min_separation=grouping_separation), aperture_radius=aperture_radius, progress_bar=False,
                    xy_bounds=(aperture_radius, aperture_radius))
    photometry_object = PSFPhotometry(psf_model, fit_shape, **_accepted_keywords(PSFPhotometry, keywords))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return photometry_object(data, error=error_image, mask=unusable, init_params=initial)


def iterative_psf_photometry(image, psf_model, sources, stamp_size, fit_shape_in_fwhm=2.0, grouping_separation_in_fwhm=2.0,
                             number_of_iterations=3, residual_threshold_in_sigmas=5.0, psf_model_error_fraction=0.05, psf_star_table=None,
                             psf_oversampling=2, verbose=True):

    """
    PSF photometry with iterative detection (as DAOPHOT's ALLSTAR + FIND on the subtracted image):
        1. fit all detected stars (overlapping stars are fitted simultaneously in groups);
        2. subtract the fitted stars and look in the residual image for stars missed by the detection (close companions, stars in the
           wings of bright stars). The detection uses a noise map that includes, besides the pixel variance, an uncertainty of
           psf_model_error_fraction x the PSF model, so that the residuals of imperfectly subtracted bright stars are not taken as stars;
        3. fit again all stars (old and new) and repeat.
    After each fit (except the last), the empirical PSF is rebuilt from its stars with the fitted neighbours subtracted (if psf_star_table
    is given). Returns the final photutils table, the model image of all stars, the number of stars added per iteration and the final PSF.
    """

    fwhm = image.fwhm
    fit_size = int(2*np.ceil(0.5*fit_shape_in_fwhm*fwhm) + 1)
    fit_shape = (fit_size, fit_size)
    error_image = np.sqrt(np.clip(np.where(np.isfinite(image.variance), image.variance, np.inf), 1e-6, None))
    positions = np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)]
    fluxes = np.clip(np.asarray(sources["flux_estimate"], dtype=float), 1.0, None)
    added = []
    result, model_image = None, None
    previous_iteration = np.zeros(0, dtype=int)       # iteration in which each star of the current list was detected
    for iteration in range(number_of_iterations):
        grouping_separation = adaptive_grouping_separation(positions, grouping_separation_in_fwhm*fwhm, fwhm)
        result = fit_psf_photometry(image, psf_model, positions, fluxes, fit_shape, error_image, grouping_separation, fwhm)
        result["iteration_detected"] = np.r_[previous_iteration, np.full(len(result) - len(previous_iteration), iteration)]
        good = np.isfinite(np.asarray(result["flux_fit"], dtype=float)) & (np.asarray(result["flux_fit"], dtype=float) > 0)
        fitted_positions = np.c_[np.asarray(result["x_fit"], dtype=float), np.asarray(result["y_fit"], dtype=float)]
        fitted_fluxes = np.asarray(result["flux_fit"], dtype=float)
        model_image = render_stars(psf_model, fitted_positions[good], fitted_fluxes[good], image.data.shape, stamp_size)
        if iteration == number_of_iterations - 1:
            break
        residual = image.data - image.background - model_image
        if psf_star_table is not None:
            psf_model = rebuild_empirical_psf(image, psf_star_table, stamp_size, residual, fitted_positions[good], fitted_fluxes[good],
                                              psf_model, oversampling=psf_oversampling)
        noise_map = np.sqrt(error_image**2 + (psf_model_error_fraction*model_image)**2)
        # No new stars where the model cannot be trusted: around saturated pixels (their residuals look like stars)
        saturated = image.data > image.saturation_level
        if np.any(saturated):
            from scipy import ndimage
            noise_map = np.where(ndimage.binary_dilation(saturated, iterations=int(np.ceil(3*fwhm))), np.inf, noise_map)
        new_sources = aux.detect_stars(image, threshold_in_sigmas=residual_threshold_in_sigmas, fwhm=fwhm, minimum_separation=1.0,
                                       data=residual, threshold_image=noise_map)
        if len(new_sources):
            new_positions = np.c_[np.asarray(new_sources["x"], dtype=float), np.asarray(new_sources["y"], dtype=float)]
            # Not on top of a star that is already fitted
            distance, _ = aux.cKDTree(fitted_positions[good]).query(new_positions) if np.any(good) else (np.full(len(new_positions),
                                                                                                                 np.inf), None)
            new_positions = new_positions[distance > 1.0*fwhm]
        else:
            new_positions = np.zeros((0, 2))
        added.append(len(new_positions))
        if verbose:
            print(f"   iteration {iteration + 1}: {len(result)} stars fitted (grouping distance {grouping_separation/fwhm:.2f} FWHM), "
                  f"{len(new_positions)} new stars found in the residual image")
        if len(new_positions) == 0:
            break
        new_fluxes = np.clip(aux.quick_aperture_fluxes(image, new_positions, fwhm, data=residual), 1.0, None)*2.0
        keep = good
        previous_iteration = np.asarray(result["iteration_detected"])[keep]
        positions = np.r_[fitted_positions[keep], new_positions]
        fluxes = np.r_[fitted_fluxes[keep], new_fluxes]
    result = merge_duplicate_stars(result, 0.5*fwhm)
    good = np.isfinite(np.asarray(result["flux_fit"], dtype=float)) & (np.asarray(result["flux_fit"], dtype=float) > 0)
    model_image = render_stars(psf_model, np.c_[np.asarray(result["x_fit"], dtype=float), np.asarray(result["y_fit"], dtype=float)][good],
                               np.asarray(result["flux_fit"], dtype=float)[good], image.data.shape, stamp_size)
    return result, model_image, added, psf_model


def merge_duplicate_stars(result, merge_distance):
    """
    Two fitted components that converged to (almost) the same position are the same star split in two: they are merged (fluxes added,
    position of the brighter one, errors added in quadrature).
    """
    positions = np.c_[np.asarray(result["x_fit"], dtype=float), np.asarray(result["y_fit"], dtype=float)]
    fluxes = np.asarray(result["flux_fit"], dtype=float)
    valid = np.all(np.isfinite(positions), axis=1) & np.isfinite(fluxes)
    if np.count_nonzero(valid) < 2:
        return result
    tree = aux.cKDTree(positions[valid])
    valid_index = np.flatnonzero(valid)
    removed = np.zeros(len(result), dtype=bool)
    flux_column = np.array(result["flux_fit"], dtype=float)
    error_column = np.array(result["flux_err"], dtype=float)
    for first, second in sorted(tree.query_pairs(merge_distance)):
        first, second = valid_index[first], valid_index[second]
        if removed[first] or removed[second]:
            continue
        keep, drop = (first, second) if fluxes[first] >= fluxes[second] else (second, first)
        flux_column[keep] = flux_column[keep] + flux_column[drop]
        error_column[keep] = np.hypot(error_column[keep], error_column[drop])
        removed[drop] = True
    result["flux_fit"] = flux_column
    result["flux_err"] = error_column
    return result[~removed]


def psf_to_total_correction(image, psf_model, stamp_size, positions, psf_fluxes, psf_flux_errors, flags, model_image, large_radius,
                            maximum_number_of_stars=20):

    """
    Ratio between the total flux (aperture of large_radius, ~4 FWHM) and the PSF flux of bright isolated stars. The ePSF is normalized
    within its (finite) stamp, so this correction brings the PSF fluxes to the same scale as the aperture-corrected fluxes (and as the
    standard-star calibration). Only isolated stars are used (no fitted neighbour brighter than 1% within large_radius + 2 FWHM, relaxed
    if needed), and the remaining neighbours are subtracted with their PSF models before the aperture is measured.
    Returns (correction, uncertainty, number of stars, intrinsic scatter).
    """

    unflagged = (flags == 0) & np.all(np.isfinite(positions), axis=1) & (psf_fluxes > 0)
    # Isolated stars only: even subtracted, the neighbours (and the unresolved stars around them) leave residual light in the large
    # aperture. The isolation criterion is relaxed if there are fewer than 5 such stars.
    fwhm = image.fwhm or 3.0
    usable_positions = np.where(np.isfinite(positions), positions, -1e6)
    for maximum_ratio in (0.01, 0.03, 0.1):
        isolated = aux.isolation_mask(usable_positions, np.clip(psf_fluxes, 0, None), usable_positions[unflagged],
                                      np.clip(psf_fluxes[unflagged], 0, None), large_radius + 2*fwhm, maximum_ratio)
        if np.count_nonzero(unflagged & isolated) >= 5:
            break
    candidates = unflagged & isolated
    order = np.argsort(-np.where(candidates, psf_fluxes, -np.inf))[:4*maximum_number_of_stars]
    order = [index for index in order if candidates[index]]
    if len(order) < 3:
        return 1.0, np.nan, 0, 0.0
    candidate_positions = positions[order]
    residual = image.data - image.background - model_image
    # Aperture sums of (residual + own model) = image with all the other stars removed; the own model is added back star by star
    residual_sums = np.asarray(aperture_photometry(np.where(image.bad_pixels, 0.0, residual), CircularAperture(candidate_positions,
                                                                                                          large_radius))["aperture_sum"])
    own_sums = np.empty(len(order))
    single = psf_model.copy()
    integer_radius = int(np.ceil(large_radius)) + 1
    for index, ((x, y), flux) in enumerate(zip(candidate_positions, psf_fluxes[order])):
        y_grid, x_grid = np.mgrid[int(y) - integer_radius:int(y) + integer_radius + 2, int(x) - integer_radius:int(x) + integer_radius + 2]
        single.x_0, single.y_0, single.flux = x, y, flux
        stamp = single(x_grid, y_grid)
        own_sums[index] = aperture_photometry(stamp, CircularAperture([(x - x_grid[0, 0], y - y_grid[0, 0])], large_radius))["aperture_sum"][0]
    large_aperture_flux = residual_sums + own_sums
    variance_sums = np.asarray(aperture_photometry(np.clip(np.where(np.isfinite(image.variance), image.variance, 0.0), 0, None),
                                                   CircularAperture(candidate_positions, large_radius))["aperture_sum"])
    ratio = large_aperture_flux/psf_fluxes[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        signal_to_noise = large_aperture_flux/np.sqrt(variance_sums)
    usable = np.isfinite(ratio) & (ratio > 0.7) & (ratio < 1.5) & (signal_to_noise > 50)
    if np.count_nonzero(usable) < 3:
        return 1.0, np.nan, 0, 0.0
    return aux.robust_ratio_correction(ratio, psf_fluxes[order], psf_flux_errors[order], np.sqrt(variance_sums), usable,
                                       maximum_number_of_stars)
