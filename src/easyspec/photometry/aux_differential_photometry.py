"""
Auxiliary functions of easyspec.photometry for differential photometry of time series (e.g. exoplanet transits): registration of the
frames, measurement of the target and of the comparison stars in every frame, choice of the comparison ensemble and of the aperture,
and the time stamps (BJD_TDB).

The frames are not resampled (no alignment): the stars are followed as they drift on the detector, which keeps the noise of each pixel
independent and the variance from the VAR extension exact.
"""

import warnings

import numpy as np
from scipy import ndimage

from . import aux_photometry as aux


# ======================================================================================================================================
# Time stamps
# ======================================================================================================================================

def observatory_location(header):
    """EarthLocation from the usual header keywords (LATITUDE/LONGITUD/HEIGHT, SITELAT/SITELONG/SITEELEV, OBSGEO-*), or None."""
    from astropy import units as u
    from astropy.coordinates import Angle, EarthLocation
    try:
        if all(keyword in header for keyword in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z")):
            return EarthLocation.from_geocentric(header["OBSGEO-X"], header["OBSGEO-Y"], header["OBSGEO-Z"], unit=u.m)
        for latitude_key, longitude_key, height_key in (("LATITUDE", "LONGITUD", "HEIGHT"), ("SITELAT", "SITELONG", "SITEELEV"),
                                                         ("LAT-OBS", "LONG-OBS", "ALT-OBS")):
            if latitude_key in header and longitude_key in header:
                latitude = Angle(header[latitude_key], unit=u.deg)
                longitude = Angle(header[longitude_key], unit=u.deg)
                return EarthLocation(lat=latitude, lon=longitude, height=float(header.get(height_key, 0.0))*u.m)
    except Exception:
        return None
    return None


def mid_exposure_times(images, target_coordinates=None, time_scale="auto"):
    """
    Mid-exposure times of the frames. time_scale = "bjd_tdb" (barycentric Julian date, needs the target coordinates and the observatory
    location in the header), "mjd_utc", "jd_utc", or "auto" (BJD_TDB when possible, otherwise MJD UTC). Returns (times, scale name).
    """
    from astropy import units as u
    from astropy.time import Time
    starts = []
    for image in images:
        if "DATE-OBS" in image.header and "T" in str(image.header["DATE-OBS"]):
            starts.append(Time(image.header["DATE-OBS"], scale="utc"))
        elif image.mjd is not None:
            starts.append(Time(image.mjd, format="mjd", scale="utc"))
        else:
            raise ValueError(f"{image.name}: no DATE-OBS or MJD-OBS in the header.")
    exposure = np.array([image.exposure_time or 0.0 for image in images])
    middle = Time([start.mjd for start in starts], format="mjd", scale="utc") + 0.5*exposure*u.s
    location = observatory_location(images[0].header)
    if time_scale in ("auto", "bjd_tdb") and target_coordinates is not None and location is not None:
        middle = Time(middle.mjd, format="mjd", scale="utc", location=location)
        barycentric = middle.tdb + middle.light_travel_time(target_coordinates)
        return barycentric.jd, "BJD_TDB"
    if time_scale == "bjd_tdb":
        raise ValueError("BJD_TDB needs the target coordinates (WCS) and the observatory location in the header.")
    if time_scale == "jd_utc":
        return middle.jd, "JD_UTC"
    return middle.mjd, "MJD_UTC"


# ======================================================================================================================================
# Measurements in each frame
# ======================================================================================================================================

def recentre(data, x, y, half_size):
    """Intensity-weighted centroid in a box (background = median of the box), iterated twice."""
    for _ in range(2):
        x_center, y_center = int(round(x)), int(round(y))
        box = data[y_center - half_size:y_center + half_size + 1, x_center - half_size:x_center + half_size + 1]
        if box.shape != (2*half_size + 1, 2*half_size + 1) or not np.all(np.isfinite(box)):
            return np.nan, np.nan
        weights = np.clip(box - np.median(box), 0, None)
        if weights.sum() <= 0:
            return np.nan, np.nan
        y_grid, x_grid = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
        x, y = x_center + np.sum(weights*x_grid)/weights.sum(), y_center + np.sum(weights*y_grid)/weights.sum()
    return x, y


def measure_frame(image, predicted_positions, aperture_radii, annulus_radii, centroid_half_size, number_of_bright_stars=8):
    """
    Fluxes of the stars of one frame in several apertures. The positions are the predicted ones shifted by the median centroid offset of the
    brightest stars (a common shift is more precise than individual centroids for faint stars). Returns positions, fluxes, errors (stars x
    radii), sky levels and peak values.
    """
    number_of_rows, number_of_columns = image.data.shape
    margin = annulus_radii[1] + 2
    centroids = np.array([recentre(image.data, x, y, centroid_half_size) for x, y in predicted_positions])
    offsets = centroids - predicted_positions
    common_offset = np.nanmedian(offsets[:number_of_bright_stars], axis=0)
    positions = predicted_positions + np.nan_to_num(common_offset)
    inside = ((positions[:, 0] > margin) & (positions[:, 0] < number_of_columns - margin) & (positions[:, 1] > margin)
              & (positions[:, 1] < number_of_rows - margin))
    number_of_stars = len(positions)
    fluxes = np.full((number_of_stars, len(aperture_radii)), np.nan)
    errors = np.full_like(fluxes, np.nan)
    sky_level = np.full(number_of_stars, np.nan)
    peaks = np.full(number_of_stars, np.nan)
    if not np.any(inside):
        return np.where(inside[:, None], positions, np.nan), fluxes, errors, sky_level, peaks
    measured = positions[inside]
    sky, sky_std, number_of_sky_pixels = aux.local_sky_in_annuli(image, measured, annulus_radii[0], annulus_radii[1])
    for radius_index, radius in enumerate(aperture_radii):
        flux, error, _ = aux.aperture_fluxes(image, measured, radius, sky, sky_std, number_of_sky_pixels)
        fluxes[inside, radius_index] = flux
        errors[inside, radius_index] = error
    sky_level[inside] = sky
    for index in np.flatnonzero(inside):
        x_center, y_center = int(round(positions[index, 0])), int(round(positions[index, 1]))
        peaks[index] = np.nanmax(image.data[y_center - 2:y_center + 3, x_center - 2:x_center + 3])
    return np.where(inside[:, None], positions, np.nan), fluxes, errors, sky_level, peaks


# ======================================================================================================================================
# Light curve
# ======================================================================================================================================

def point_to_point_scatter(values):
    """Robust point-to-point scatter of a time series (insensitive to slow trends and to the transit itself)."""
    values = np.asarray(values)[np.isfinite(values)]
    if len(values) < 6:
        return np.inf
    differences = np.diff(values)
    return 1.4826*np.median(np.abs(differences - np.median(differences)))/np.sqrt(2)


def comparison_systematics(fluxes, frames, comparisons, bin_size=10):
    """
    Median excess scatter (beyond white noise) of bins of bin_size frames of each comparison star divided by the other comparisons: the
    level of slow systematics (flat-field errors as the stars drift, non-uniform clouds, ...) that a light curve of this field can have.
    """
    excess = []
    for star in comparisons:
        others = [other for other in comparisons if other != star]
        if not others:
            continue
        ensemble = np.sum(fluxes[:, others]/np.nanmedian(fluxes[frames][:, others], axis=0), axis=1)
        relative = (fluxes[:, star]/ensemble)[frames]
        relative = relative/np.nanmedian(relative)
        number_of_bins = len(relative)//bin_size
        if number_of_bins < 3:
            continue
        binned = relative[:number_of_bins*bin_size].reshape(number_of_bins, bin_size).mean(axis=1)
        excess.append(np.sqrt(max(np.nanstd(binned)**2 - point_to_point_scatter(relative)**2/bin_size, 0.0)))
    return float(np.median(excess)) if excess else np.nan


def build_differential_light_curve(fluxes, errors, peaks, saturation_level, target_index=0, comparison_indices=None,
                                   minimum_presence=0.9, minimum_relative_brightness=0.1, minimum_transparency=0.7, jump_threshold=5.0,
                                   outlier_threshold=5.0, rejection_threshold=2.0, reject_noisy_comparisons=True):

    """
    Choose the frames, the comparison stars and the aperture radius, and compute the differential light curve of the target.

    fluxes, errors: arrays (frames x stars x radii). For each aperture radius:
        - comparisons: stars measured in at least minimum_presence of the frames, never above 85% of saturation, and brighter than
          minimum_relative_brightness x the target (or the given comparison_indices);
        - frames: target and all comparisons measured, and transparency (summed flux of the comparisons) above minimum_transparency
          of its median (clouds);
        - each comparison is compared with the ensemble of the others, and the noisiest (point-to-point scatter more than
          rejection_threshold x the median, relative to its photon noise) is removed, iteratively (variable stars, bad pixels), unless
          reject_noisy_comparisons=False (then all comparisons are kept, and the noise ratio of each one is only reported);
        - frames where one comparison jumps (cosmic ray, satellite) by more than jump_threshold sigma from its running median are dropped;
        - the ensemble is the weighted sum of the normalized comparisons (weights = inverse relative variance).
    The aperture with the lowest point-to-point scatter of the target light curve is chosen (this criterion is not affected by the
    transit). Finally, outliers (outlier_threshold sigma from a running median) are flagged. Returns a dict.
    """

    number_of_frames, number_of_stars, number_of_radii = fluxes.shape
    middle_radius = number_of_radii//2
    target_measured = np.isfinite(fluxes[:, target_index, middle_radius])
    if comparison_indices is None:
        presence = np.mean(np.isfinite(fluxes[target_measured, :, middle_radius]), axis=0)
        unsaturated = np.nanmax(np.where(np.isfinite(peaks[target_measured]), peaks[target_measured], 0), axis=0) < saturation_level
        brightness = np.nanmedian(fluxes[target_measured, :, middle_radius], axis=0)
        candidates = [star for star in range(number_of_stars) if star != target_index and presence[star] >= minimum_presence
                      and unsaturated[star] and brightness[star] > minimum_relative_brightness*brightness[target_index]]
    else:
        candidates = [int(star) for star in comparison_indices if star != target_index]
    if len(candidates) < 2:
        raise RuntimeError(f"Only {len(candidates)} comparison stars are measured in (almost) all frames. Use a larger field, fainter "
                           "comparisons (minimum_relative_brightness) or give comparison_indices.")
    frames = target_measured & np.all(np.isfinite(fluxes[:, candidates, middle_radius]), axis=1)
    transparency = np.nansum(fluxes[:, candidates, middle_radius], axis=1)
    transparency = transparency/np.nanmedian(transparency[frames])
    frames &= transparency > minimum_transparency
    if np.count_nonzero(frames) < 10:
        raise RuntimeError("Fewer than 10 usable frames (target and comparisons measured, transparency above the threshold).")

    best = None
    for radius_index in range(number_of_radii):
        flux, error = fluxes[:, :, radius_index], errors[:, :, radius_index]
        comparisons = list(candidates)
        for _ in range(len(candidates)):
            ensemble = np.nansum(flux[:, comparisons], axis=1)
            noise_ratio = []
            for star in comparisons:
                ratio = flux[frames, star]/(ensemble[frames] - flux[frames, star])
                expected = np.nanmedian(error[frames, star]/flux[frames, star])
                noise_ratio.append(point_to_point_scatter(ratio/np.nanmedian(ratio))/expected)
            worst = int(np.argmax(noise_ratio))
            if reject_noisy_comparisons and len(comparisons) > 2 and noise_ratio[worst] > rejection_threshold*np.median(noise_ratio):
                comparisons.pop(worst)
            else:
                break
        # Frames where one comparison jumps
        ensemble = np.nansum(flux[:, comparisons], axis=1)
        jumps = np.zeros(number_of_frames, dtype=bool)
        for star in comparisons:
            ratio = flux[:, star]/(ensemble - flux[:, star])
            ratio = ratio/np.nanmedian(ratio[frames])
            running = ndimage.median_filter(np.where(frames, ratio, 1.0), 9, mode="nearest")
            sigma = 1.4826*np.nanmedian(np.abs((ratio - running)[frames]))
            jumps |= np.abs(ratio - running) > jump_threshold*sigma
        used_frames = frames & ~jumps
        reference_flux = np.nanmedian(flux[used_frames][:, comparisons], axis=0)
        weights = 1/np.nanmedian((error[used_frames][:, comparisons]/flux[used_frames][:, comparisons])**2, axis=0)
        weights = weights/weights.sum()
        ensemble = np.sum(flux[:, comparisons]/reference_flux*weights, axis=1)
        ensemble_relative_error = np.sqrt(np.sum((weights*error[:, comparisons]/reference_flux)**2, axis=1))/ensemble
        relative_flux = flux[:, target_index]/ensemble
        relative_error = np.abs(relative_flux)*np.hypot(error[:, target_index]/flux[:, target_index], ensemble_relative_error)
        normalization = np.nanmedian(relative_flux[used_frames])
        scatter = point_to_point_scatter(relative_flux[used_frames]/normalization)
        if best is None or scatter < best["point_to_point_scatter"]:
            best = dict(radius_index=radius_index, comparisons=comparisons, weights=weights, relative_flux=relative_flux/normalization,
                        comparison_noise_ratio=dict(zip(comparisons, np.round(noise_ratio, 2).tolist())) if len(noise_ratio) == len(comparisons)
                        else {},
                        relative_flux_error=relative_error/normalization, frames=used_frames, point_to_point_scatter=scatter,
                        candidates=candidates)
    # Outliers from a running median
    relative_flux = best["relative_flux"]
    filled = np.where(best["frames"], relative_flux, np.nanmedian(relative_flux[best["frames"]]))
    running = ndimage.median_filter(filled, 11, mode="nearest")
    sigma = 1.4826*np.nanmedian(np.abs((relative_flux - running)[best["frames"]]))
    best["outliers"] = best["frames"] & (np.abs(relative_flux - running) > outlier_threshold*sigma)
    best["frames"] = best["frames"] & ~best["outliers"]
    best["transparency"] = transparency
    best["comparison_systematics"] = comparison_systematics(fluxes[:, :, best["radius_index"]], best["frames"], best["comparisons"])
    return best


def bin_light_curve(time, flux, number_per_bin, flux_error=None):
    """
    Averages of consecutive groups of number_per_bin points: time, flux and the error of the mean (propagated from flux_error if given,
    otherwise from the scatter within the bin, which is inflated where the flux changes quickly, e.g. at ingress and egress).
    """
    number_of_bins = len(time)//number_per_bin
    if number_of_bins < 1:
        return np.array([]), np.array([]), np.array([])
    time_bins = time[:number_of_bins*number_per_bin].reshape(number_of_bins, number_per_bin)
    flux_bins = flux[:number_of_bins*number_per_bin].reshape(number_of_bins, number_per_bin)
    if flux_error is not None:
        error_bins = flux_error[:number_of_bins*number_per_bin].reshape(number_of_bins, number_per_bin)
        return time_bins.mean(axis=1), flux_bins.mean(axis=1), np.sqrt(np.sum(error_bins**2, axis=1))/number_per_bin
    return time_bins.mean(axis=1), flux_bins.mean(axis=1), flux_bins.std(axis=1, ddof=1)/np.sqrt(number_per_bin)


def warn_if_saturated(peaks, saturation_level, target_index=0):
    """Warn when the target reaches the non-linear regime in some frames."""
    target_peaks = peaks[:, target_index]
    fraction = np.mean(target_peaks[np.isfinite(target_peaks)] > saturation_level) if np.any(np.isfinite(target_peaks)) else 0.0
    if fraction > 0:
        warnings.warn(f"The target is above 85% of the saturation level in {100*fraction:.0f}% of the frames: defocus the telescope or use "
                      "shorter exposures.")
