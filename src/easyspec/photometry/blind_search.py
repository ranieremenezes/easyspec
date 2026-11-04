"""
Blind search for transiting planets in a field monitored on many nights, possibly by several observers with different telescopes, cameras
and sites (e.g. a network of amateur astronomers). Called by photometry.blind_search(); most of the work is done here.

Steps:
    1. Inventory of the frames (header only): observer/instrument, night, time, airmass.
    2. Star catalogue in sky coordinates: stars detected in the best frame of each instrument, merged by position (so that fields of view,
       pixel scales and camera orientations can differ between observers). Every frame must have a WCS (plate solution).
    3. Aperture photometry of every catalogue star in every frame (in parallel, one frame per task), at the positions given by the frame's
       WCS refined with the centroids of the brightest stars, in several apertures scaled with the FWHM of the frame. The measurements are
       cached in the output directory, so that new nights can be added later without measuring the old frames again.
    4. Light curves: instrumental magnitudes relative to each star's median for each instrument (removes the different zero points and
       color terms of the instruments), minus the zero point of each frame (transparency) measured on the brightest stars, best aperture per
       star (smallest point-to-point scatter), frames with clouds or bad seeing rejected, and the remaining systematics common to many stars
       (extinction x color, flat-field and focus effects) removed with SysRem (Tamuz, Mazeh & Zucker 2005).
    5. Box Least Squares periodogram of each star (in parallel, one star per task): period, epoch, duration, depth, signal-to-noise ratio and
       signal detection efficiency (SDE).
    6. Vetting: number of observed transits, odd/even depth difference and secondary eclipse (eclipsing binaries), sinusoidal variability,
       excessive depth, periods shared by many stars (systematics) and variability out of transit.

The result (BlindSearchResult) holds the star table, the light curves and the periodograms, and has methods to plot, save, reload and
extract the light curve of any star (e.g. for photometry.fit_transit()).
"""

import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import aux_differential_photometry as aux_differential
from . import aux_photometry as aux

MAGNITUDE_TO_RELATIVE_FLUX = 0.4*np.log(10)
DEFAULT_APERTURE_RADII = (0.8, 1.1, 1.5, 2.0)          # in units of the FWHM of each frame
DEFAULT_SKY_ANNULUS = (3.0, 5.0)                       # idem
HEADER_KEYWORDS = ("DATE-OBS", "MJD-OBS", "EXPTIME", "AIRMASS", "OBSERVER", "TELESCOP", "INSTRUME", "FILTER", "OBJECT", "SITELAT",
                   "SITELONG", "SITEELEV", "LATITUDE", "LONGITUD", "HEIGHT", "LAT-OBS", "LONG-OBS", "ALT-OBS", "OBSGEO-X", "OBSGEO-Y",
                   "OBSGEO-Z")


# ======================================================================================================================================
# Parallel execution
# ======================================================================================================================================

def number_of_workers(number_of_cores):
    """Number of processes: all cores but one by default."""
    available = os.cpu_count() or 1
    if number_of_cores is None:
        return max(available - 1, 1)
    return int(np.clip(number_of_cores, 1, available))


def parallel_map(function, tasks, number_of_cores, description="", verbose=True):
    """
    function(task) for every task, in parallel processes (results in the order of the tasks). The function must be defined at the top level
    of a module (here), so that it can be sent to the processes also when the code runs in a Jupyter notebook.
    """
    results = [None]*len(tasks)
    report_every = max(len(tasks)//10, 1)
    if number_of_cores <= 1 or len(tasks) < 2:
        for index, task in enumerate(tasks):
            results[index] = function(task)
            if verbose and description and (index + 1) % report_every == 0:
                print(f"   {description}: {index + 1}/{len(tasks)}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=number_of_cores) as executor:
        futures = {executor.submit(function, task): index for index, task in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if verbose and description and done % report_every == 0:
                print(f"   {description}: {done}/{len(tasks)}", flush=True)
    return results


# ======================================================================================================================================
# 1. Inventory of the frames
# ======================================================================================================================================

def _frame_information(source):
    """Header information of one frame (path or Image)."""
    if isinstance(source, aux.Image):
        header, name, shape = source.header, source.name, source.data.shape
    else:
        with fits.open(source) as hdu_list:
            header, shape = hdu_list[0].header, hdu_list[0].data.shape
        name = Path(source).name
    items = {key: header[key] for key in HEADER_KEYWORDS if key in header}
    instrument = " / ".join(str(header[key]).strip() for key in ("OBSERVER", "TELESCOP", "INSTRUME") if str(header.get(key, "")).strip())
    if not instrument:
        instrument = Path(source).parent.name if not isinstance(source, aux.Image) else "instrument"
    has_wcs = all(key in header for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"))
    return {"name": name, "instrument": instrument, "header_items": items, "shape": shape, "has_wcs": has_wcs}


def collect_frames(source, extensions=(".fits", ".fit", ".fts")):
    """Sorted list of frame paths from files and/or directories; directories are searched recursively (e.g. one folder per night)."""
    if isinstance(source, (str, Path)):
        source = [source]
    paths = []
    for item in source:
        item = Path(item)
        if item.is_dir():
            paths += sorted(path for path in item.rglob("*") if path.suffix.lower() in extensions and path.is_file())
        elif item.exists():
            paths.append(item)
        else:
            raise FileNotFoundError(f"{item} does not exist.")
    if not paths:
        raise FileNotFoundError(f"No frames found in {source}.")
    return paths


def mid_exposure_times(frames, coordinates):
    """
    BJD_TDB of mid-exposure (MJD UTC if the site is unknown), from DATE-OBS (or MJD-OBS) and EXPTIME. Returns the times, the time scale,
    the night of each frame (local date of the evening) and the mid-exposure MJD UTC.
    """
    from astropy.time import Time
    start_mjd = []
    for frame in frames:
        items = frame["header_items"]
        if "DATE-OBS" in items and "T" in str(items["DATE-OBS"]):
            start_mjd.append(Time(items["DATE-OBS"], scale="utc").mjd)
        elif "MJD-OBS" in items:
            start_mjd.append(float(items["MJD-OBS"]))
        else:
            raise ValueError(f"{frame['name']}: no DATE-OBS or MJD-OBS in the header.")
    exposure = np.array([float(frame["header_items"].get("EXPTIME", 0.0)) for frame in frames])
    middle_mjd = np.array(start_mjd) + 0.5*exposure/86400
    times = np.full(len(frames), np.nan)
    nights = []
    locations = [aux_differential.observatory_location(fits.Header(list(frame["header_items"].items()))) for frame in frames]
    time_scale = "BJD_TDB"
    for index, (frame, location) in enumerate(zip(frames, locations)):
        longitude = location.lon.deg if location is not None else 0.0
        night_mjd = np.floor(middle_mjd[index] + longitude/360 - 0.5)
        nights.append(Time(night_mjd, format="mjd").iso[:10])
        frame["location"] = location
    with_location = np.array([location is not None for location in locations])
    if np.all(with_location) and coordinates is not None:
        groups = {}
        for index, location in enumerate(locations):
            groups.setdefault((round(location.lat.deg, 5), round(location.lon.deg, 5), round(location.height.value, 1)), []).append(index)
        for indices in groups.values():
            time = Time(middle_mjd[indices], format="mjd", scale="utc", location=locations[indices[0]])
            times[indices] = (time.tdb + time.light_travel_time(coordinates)).jd
    else:
        warnings.warn("Site coordinates (SITELAT/SITELONG, LATITUDE/LONGITUD, OBSGEO-*) missing in some frames: the times are MJD UTC, not "
                      "barycentric. Frames of different sites can then be shifted by up to ~8 minutes.")
        times = middle_mjd
        time_scale = "MJD_UTC"
    return times, time_scale, nights, middle_mjd


def airmass_of(frames, times_mjd_utc, coordinates):
    """Airmass from the header, or computed from the site and the field coordinates."""
    airmass = np.array([float(frame["header_items"].get("AIRMASS", np.nan)) for frame in frames])
    missing = ~np.isfinite(airmass)
    if np.any(missing) and coordinates is not None:
        from astropy import units as u
        from astropy.coordinates import AltAz
        from astropy.time import Time
        for index in np.flatnonzero(missing):
            location = frames[index].get("location")
            if location is None:
                continue
            altitude = coordinates.transform_to(AltAz(obstime=Time(times_mjd_utc[index], format="mjd"), location=location)).alt
            airmass[index] = 1/np.sin(max(altitude.to_value(u.rad), 0.05))
    return airmass


# ======================================================================================================================================
# 2. Star catalogue
# ======================================================================================================================================

def _load(source):
    return source if isinstance(source, aux.Image) else aux.read_image(source)


def _background_and_fwhm(image, fwhm_guess=3.0):
    """Background (in image.background / background_rms) and FWHM (pixels) of a frame."""
    box_size = int(np.clip(min(image.data.shape)//3, 16, 64))
    image.background, image.background_rms = aux.estimate_background(image, box_size=box_size)
    source_mask = aux.source_mask_from_threshold(image.data, image.background, image.background_rms)
    image.background, image.background_rms = aux.estimate_background(image, box_size=box_size, source_mask=source_mask)
    bright = aux.detect_stars(image, threshold_in_sigmas=15.0, fwhm=fwhm_guess, maximum_number_of_sources=80)
    stamp_size = int(max(15, 2*np.ceil(3*fwhm_guess) + 1))
    fwhm, _, number = aux.measure_fwhm(image, bright, number_of_stars=20, stamp_size=stamp_size)
    if not np.isfinite(fwhm) or number < 3:
        fwhm = fwhm_guess
    return float(fwhm)


def _detect_in_frame(task):
    """Stars of one frame (for the catalogue): positions, sky coordinates, brightness, FWHM."""
    source, detection_threshold = task
    image = _load(source)
    fwhm = _background_and_fwhm(image)
    if fwhm > 5:
        fwhm = _background_and_fwhm(image, fwhm)
    image.fwhm = fwhm
    sources = aux.detect_stars(image, threshold_in_sigmas=detection_threshold, fwhm=fwhm, minimum_separation=1.5)
    x, y = np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)
    if image.wcs is None:
        raise ValueError(f"{image.name} has no WCS: plate-solve the frames (e.g. with astrometry.net) before the blind search.")
    sky = image.wcs.pixel_to_world(x, y)
    return {"name": image.name, "number": len(sources), "ra": sky.ra.deg, "dec": sky.dec.deg,
            "flux": np.asarray(sources["flux_estimate"], dtype=float), "peak": np.asarray(sources["peak"], dtype=float),
            "fwhm": fwhm, "pixel_scale": aux.pixel_scale_of(image), "saturation_level": image.saturation_level}


def build_catalogue(sources_by_instrument, detection_threshold, maximum_number_of_stars, number_of_cores, verbose):
    """
    Detects the stars in several frames of each instrument, keeps the frame with most detections as the reference of that instrument, and
    merges the lists by sky position. Returns the catalogue and the reference frame and FWHM of each instrument.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    tasks, owners = [], []
    for instrument, sources in sources_by_instrument.items():
        sample = [sources[index] for index in np.unique(np.linspace(0, len(sources) - 1, min(len(sources), 12)).astype(int))]
        tasks += [(source, detection_threshold) for source in sample]
        owners += [(instrument, source) for source in sample]
    detections = parallel_map(_detect_in_frame, tasks, number_of_cores)
    references = {}
    for (instrument, source), detection in zip(owners, detections):
        if instrument not in references or detection["number"] > references[instrument][1]["number"]:
            references[instrument] = (source, detection)
    order = sorted(references, key=lambda key: -references[key][1]["number"])
    ra, dec, magnitude, seen_by = np.array([]), np.array([]), np.array([]), []
    for instrument in order:
        detection = references[instrument][1]
        unsaturated = detection["peak"] < 0.9*detection["saturation_level"]
        instrumental = -2.5*np.log10(np.clip(detection["flux"], 1e-3, None))
        saturated_instrumental = np.where(unsaturated, instrumental, instrumental - 0.5)    # saturated stars: brightness underestimated
        if len(ra) == 0:
            ra, dec, magnitude = detection["ra"], detection["dec"], saturated_instrumental
            seen_by = [instrument]*len(ra)
            continue
        match_radius = max(2.0, 1.0*detection["fwhm"]*detection["pixel_scale"])*u.arcsec
        catalogue_coordinates = SkyCoord(ra, dec, unit="deg")
        new_coordinates = SkyCoord(detection["ra"], detection["dec"], unit="deg")
        index, separation, _ = new_coordinates.match_to_catalog_sky(catalogue_coordinates)
        matched = separation < match_radius
        both = matched & unsaturated
        offset = np.median(magnitude[index[both]] - instrumental[both]) if np.count_nonzero(both) >= 3 else 0.0
        for star in np.flatnonzero(matched):
            seen_by[index[star]] += ", " + instrument
        ra = np.r_[ra, detection["ra"][~matched]]
        dec = np.r_[dec, detection["dec"][~matched]]
        magnitude = np.r_[magnitude, saturated_instrumental[~matched] + offset]
        seen_by += [instrument]*int(np.count_nonzero(~matched))
    order_by_brightness = np.argsort(magnitude)[:maximum_number_of_stars]
    catalogue = Table({"id": np.arange(1, len(order_by_brightness) + 1), "ra": ra[order_by_brightness], "dec": dec[order_by_brightness],
                       "instrumental_magnitude": magnitude[order_by_brightness],
                       "seen_by": np.array(seen_by, dtype=object)[order_by_brightness].astype(str)})
    reference_frames = {instrument: references[instrument][0] for instrument in order}
    fwhm_by_instrument = {instrument: references[instrument][1]["fwhm"] for instrument in order}
    if verbose:
        for instrument in order:
            detection = references[instrument][1]
            print(f"   {instrument}: reference frame {detection['name']}, {detection['number']} stars, FWHM {detection['fwhm']:.2f} px "
                  f"({detection['fwhm']*detection['pixel_scale']:.1f}\")")
        print(f"   catalogue: {len(catalogue)} stars")
    return catalogue, reference_frames, fwhm_by_instrument


# ======================================================================================================================================
# 3. Photometry of every frame
# ======================================================================================================================================

def _measure_frame(task):
    """Fluxes of all catalogue stars in one frame (several apertures scaled with the FWHM of the frame)."""
    source, ra, dec, fwhm_guess, aperture_radii, sky_annulus, cache_file = task
    if cache_file is not None and Path(cache_file).exists():
        with np.load(cache_file) as cached:
            return {key: cached[key] for key in cached.files}
    from astropy.coordinates import SkyCoord
    image = _load(source)
    if image.wcs is None:
        raise ValueError(f"{image.name} has no WCS: plate-solve the frames (e.g. with astrometry.net) before the blind search.")
    fwhm = _background_and_fwhm(image, fwhm_guess)
    x, y = image.wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    predicted = np.c_[x, y]
    radii = np.maximum(np.array(aperture_radii)*fwhm, 1.5)
    annulus = (max(sky_annulus[0]*fwhm, radii[-1] + 2), max(sky_annulus[1]*fwhm, radii[-1] + 6))
    # stars outside the frame are skipped by measure_frame(); the common offset uses the brightest stars inside the frame
    inside = (x > 0) & (x < image.data.shape[1] - 1) & (y > 0) & (y < image.data.shape[0] - 1)
    order = np.r_[np.flatnonzero(inside), np.flatnonzero(~inside)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        positions, fluxes, errors, sky, peaks = aux_differential.measure_frame(image, predicted[order], radii, annulus,
                                                                               centroid_half_size=max(2, int(round(1.2*fwhm))),
                                                                               number_of_bright_stars=10)
    restore = np.argsort(order)
    fluxes, errors = fluxes[restore], errors[restore]
    # Stars whose aperture contains several pixels replaced by the cosmic-ray cleaning are rejected in this frame: LACosmic sometimes takes
    # the core of a star for a cosmic ray, and the replaced pixels then remove a large part of its flux (a false "transit" of one point)
    cosmic_ray_pixels = count_flagged_pixels((image.mask & 1).astype(bool), positions[restore], radii[-1])
    replaced = cosmic_ray_pixels >= 2
    fluxes[replaced], errors[replaced] = np.nan, np.nan
    result = {"fluxes": fluxes.astype(np.float32), "errors": errors.astype(np.float32), "cosmic_ray_rejections": np.int32(replaced.sum()),
              "peaks": peaks[restore].astype(np.float32), "sky": sky[restore].astype(np.float32),
              "x": positions[restore, 0].astype(np.float32), "y": positions[restore, 1].astype(np.float32),
              "fwhm": np.float32(fwhm), "saturation_level": np.float64(image.saturation_level),
              "background": np.float32(np.median(image.background))}
    if cache_file is not None:
        np.savez(cache_file, **result)
    return result


def count_flagged_pixels(flagged, positions, radius):
    """Number of flagged pixels (boolean image) within 'radius' of each position (NaN positions give 0)."""
    counts = np.zeros(len(positions), dtype=int)
    if not np.any(flagged):
        return counts
    half = int(np.ceil(radius))
    rows, columns = flagged.shape
    for index, (x, y) in enumerate(positions):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        x_center, y_center = int(round(x)), int(round(y))
        y_start, y_end = max(y_center - half, 0), min(y_center + half + 1, rows)
        x_start, x_end = max(x_center - half, 0), min(x_center + half + 1, columns)
        if y_start >= y_end or x_start >= x_end:
            continue
        y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
        inside = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        counts[index] = int(np.count_nonzero(flagged[y_start:y_end, x_start:x_end] & inside))
    return counts


def catalogue_hash(catalogue, aperture_radii, sky_annulus):
    """Identifies the measurements made with a given catalogue and apertures (cache)."""
    text = np.array2string(np.round(np.c_[catalogue["ra"], catalogue["dec"]], 7), threshold=10**9) + str(aperture_radii) + str(sky_annulus)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


# ======================================================================================================================================
# 4. Light curves
# ======================================================================================================================================

def robust_standard_deviation(values, axis=None):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    return 1.4826*np.nanmedian(np.abs(values - median), axis=axis)


def point_to_point_scatter_by_night(residuals, nights):
    """Robust scatter of the differences between consecutive points of the same night, divided by sqrt(2) (per star; residuals: stars x frames)."""
    differences = []
    for night in np.unique(nights):
        columns = np.flatnonzero(nights == night)
        if len(columns) > 2:
            differences.append(np.diff(residuals[:, columns], axis=1))
    differences = np.concatenate(differences, axis=1)
    return 1.4826*np.nanmedian(np.abs(differences), axis=1)/np.sqrt(2)


def sysrem(residuals, errors, number_of_components, star_weights, iterations=15, clip=4.0):
    """
    SysRem (Tamuz, Mazeh & Zucker 2005): removes from residuals (stars x frames, magnitudes) the effects common to many stars, each one
    modelled as c_star x a_frame (e.g. extinction x color). NaN = missing point. Returns the cleaned residuals and the components.

    The frame coefficients a_frame are computed only with the stars of weight > 0 (star_weights: 0 for variable or saturated stars), and
    without their outlying points (more than 'clip' robust sigmas from zero): otherwise a single star with deep eclipses would dominate the
    frames of its eclipses, and SysRem would remove the eclipses from it (and add them, weakened, to all the other stars).
    """
    finite = np.isfinite(residuals) & np.isfinite(errors) & (errors > 0)
    weights = np.where(finite, 1/np.where(finite, errors, 1.0)**2, 0.0)
    cleaned = np.where(finite, residuals, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        star_scatter = robust_standard_deviation(np.where(finite, residuals, np.nan), axis=1)
    inliers = np.abs(cleaned) < clip*np.where(np.isfinite(star_scatter), star_scatter, np.inf)[:, None]
    frame_weights = weights*star_weights[:, None]*inliers
    components = []
    for _ in range(number_of_components):
        frame_coefficients = np.ones(residuals.shape[1])
        for _ in range(iterations):
            star_coefficients = (cleaned*weights) @ frame_coefficients/np.maximum(weights @ frame_coefficients**2, 1e-30)
            frame_coefficients = (star_coefficients @ (cleaned*frame_weights))/np.maximum((star_coefficients**2) @ frame_weights, 1e-30)
        star_coefficients = (cleaned*weights*inliers) @ frame_coefficients/np.maximum((weights*inliers) @ frame_coefficients**2, 1e-30)
        cleaned = cleaned - np.outer(star_coefficients, frame_coefficients)
        components.append((star_coefficients, frame_coefficients))
    return np.where(finite, cleaned, np.nan), components


def build_light_curves(measurements, frames, instruments, nights, number_of_systematics, minimum_transparency, maximum_fwhm_factor,
                       verbose):
    """
    Relative light curves (stars x frames) from the measurements: per-instrument normalization, frame zero points, best aperture per star,
    frame rejection and SysRem. Returns a dict with the residual magnitudes, errors, chosen aperture, frame quality and star statistics.
    """
    fluxes = np.stack([measurement["fluxes"] for measurement in measurements]).astype(float)      # frames x stars x radii
    errors = np.stack([measurement["errors"] for measurement in measurements]).astype(float)
    peaks = np.stack([measurement["peaks"] for measurement in measurements]).astype(float)        # frames x stars
    saturation = np.array([measurement["saturation_level"] for measurement in measurements])
    fwhm = np.array([float(measurement["fwhm"]) for measurement in measurements])
    number_of_frames, number_of_stars, number_of_radii = fluxes.shape
    saturated = peaks >= 0.9*saturation[:, None]
    with np.errstate(all="ignore"):
        magnitudes = np.where((fluxes > 0) & ~saturated[:, :, None], -2.5*np.log10(fluxes), np.nan)
        magnitude_errors = 2.5/np.log(10)*errors/fluxes
    magnitude_errors[~np.isfinite(magnitudes)] = np.nan
    instrument_names = np.unique(instruments)
    # 4a. Each star relative to its median with each instrument
    relative = np.full_like(magnitudes, np.nan)
    for instrument in instrument_names:
        rows = instruments == instrument
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relative[rows] = magnitudes[rows] - np.nanmedian(magnitudes[rows], axis=0, keepdims=True)
    # 4b. Zero point of each frame (transparency) from the brightest unsaturated stars measured in most frames of the instrument
    middle_radius = number_of_radii//2
    zero_points = np.full((number_of_frames, number_of_radii), np.nan)
    transparency = np.full(number_of_frames, np.nan)
    for instrument in instrument_names:
        rows = np.flatnonzero(instruments == instrument)
        coverage = np.mean(np.isfinite(magnitudes[rows, :, middle_radius]), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            typical = np.nanmedian(magnitudes[rows, :, middle_radius], axis=0)
        candidates = np.flatnonzero((coverage > 0.8) & np.isfinite(typical) & (np.mean(saturated[rows], axis=0) < 0.01))
        reference_stars = candidates[np.argsort(typical[candidates])][:40]
        if len(reference_stars) < 3:
            raise RuntimeError(f"{instrument}: fewer than 3 unsaturated stars measured in most frames; cannot compute the frame zero points.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero_points[rows] = np.nanmedian(relative[rows][:, reference_stars, :], axis=1)
            flux_level = 10**(-0.4*zero_points[rows, middle_radius])
            transparency[rows] = flux_level/np.nanpercentile(flux_level, 90)
    residuals_by_radius = relative - zero_points[:, None, :]
    # 4c. Frame rejection: thick clouds, bad seeing, failed measurements
    used = np.isfinite(transparency) & (transparency >= minimum_transparency)
    for instrument in instrument_names:
        rows = instruments == instrument
        used[rows] &= fwhm[rows] <= maximum_fwhm_factor*np.nanmedian(fwhm[rows])
    measured_fraction = np.mean(np.isfinite(residuals_by_radius[:, :, middle_radius]), axis=1)
    used &= measured_fraction >= 0.3*np.nanmedian(measured_fraction)
    # 4d. Best aperture per star: smallest point-to-point scatter within the nights
    scatter = np.full((number_of_stars, number_of_radii), np.inf)
    for radius_index in range(number_of_radii):
        values = residuals_by_radius[used, :, radius_index].T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scatter[:, radius_index] = point_to_point_scatter_by_night(values, nights[used])
    scatter[~np.isfinite(scatter)] = np.inf
    best_radius = np.argmin(scatter, axis=1)
    star_index = np.arange(number_of_stars)
    residuals = residuals_by_radius[:, star_index, best_radius].T          # stars x frames
    residual_errors = magnitude_errors[:, star_index, best_radius].T
    residuals[:, ~used] = np.nan
    # 4e. SysRem, with the variable and badly measured stars excluded from the frame coefficients
    coverage = np.mean(np.isfinite(residuals[:, used]), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initial_scatter = robust_standard_deviation(residuals, axis=1)
        expected = np.nanmedian(residual_errors, axis=1)
    good_for_systematics = (coverage > 0.5) & np.isfinite(initial_scatter) & (initial_scatter < 3*np.maximum(expected, 1e-4))
    star_weights = good_for_systematics.astype(float)
    if np.count_nonzero(good_for_systematics) < 5:
        warnings.warn("Fewer than 5 well-measured stars: SysRem is not applied.")
        cleaned = residuals
    else:
        cleaned, _ = sysrem(residuals, residual_errors, number_of_systematics, star_weights)
    # 4f. Outliers: only points brighter than 5 sigma (a transit or eclipse makes the star fainter, never brighter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scatter_after = robust_standard_deviation(cleaned, axis=1)
        cleaned[cleaned < -5*scatter_after[:, None]] = np.nan
        scatter_after = robust_standard_deviation(cleaned, axis=1)
        point_to_point = point_to_point_scatter_by_night(cleaned[:, used], nights[used])
    residual_errors[~np.isfinite(cleaned)] = np.nan
    saturated_fraction = np.mean(saturated, axis=0)
    if verbose:
        print(f"   {np.count_nonzero(used)} of {number_of_frames} frames used ({np.count_nonzero(~used)} rejected: clouds, seeing or failed "
              f"measurement); SysRem with {number_of_systematics} components on {np.count_nonzero(good_for_systematics)} stars")
    return {"residuals": cleaned, "errors": residual_errors, "best_radius": best_radius, "used": used, "transparency": transparency,
            "fwhm": fwhm, "scatter": scatter_after, "point_to_point": point_to_point, "expected": expected,
            "saturated_fraction": saturated_fraction}


# ======================================================================================================================================
# 5. Box Least Squares search
# ======================================================================================================================================

def red_noise_factor(time, flux, points_per_group=12, night_gap=0.25):
    """
    beta = (scatter of the means of groups of consecutive points of the same night) / (expected for white noise): the factor by which
    correlated noise (residual systematics on time scales of ~1 hour) increases the uncertainty of a transit depth (Pont et al. 2006).
    """
    nights = np.r_[0, np.cumsum(np.diff(time) > night_gap)]
    point_scatter = robust_standard_deviation(flux)
    means = []
    for night in np.unique(nights):
        values = flux[nights == night]
        for start in range(0, len(values) - points_per_group + 1, points_per_group):
            means.append(np.mean(values[start:start + points_per_group]))
    if len(means) < 5 or point_scatter <= 0:
        return 1.0
    return float(max(robust_standard_deviation(means)/(point_scatter/np.sqrt(points_per_group)), 1.0))


def window_depth(flux, selection, baseline, point_scatter, beta, minimum_points=5):
    """Mean flux deficit of the selected points relative to the baseline, and its uncertainty (white noise x beta)."""
    number = int(np.count_nonzero(selection))
    if number < minimum_points:
        return np.nan, np.nan
    return float(baseline - np.mean(flux[selection])), float(point_scatter*beta/np.sqrt(number))


def _search_star(task):
    """BLS periodogram of one star, best signal and its statistics (relative flux)."""
    star_index, time, flux, flux_error, minimum_period, maximum_period, durations, frequency_factor, minimum_points_per_transit = task
    from astropy.timeseries import BoxLeastSquares
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BoxLeastSquares(time, flux, flux_error)
        periods = np.sort(model.autoperiod(durations, minimum_period=minimum_period, maximum_period=maximum_period,
                                           frequency_factor=frequency_factor))
        periodogram = model.power(periods, durations, objective="likelihood")
    power = np.asarray(periodogram.power, dtype=float)
    # Signal detection efficiency on the periodogram minus its slowly varying trend (the window function of the nights)
    number_of_blocks = 200
    block_edges = np.linspace(0, len(power), number_of_blocks + 1).astype(int)
    block_centres = 0.5*(block_edges[:-1] + block_edges[1:])
    block_medians = np.array([np.median(power[start:end]) for start, end in zip(block_edges[:-1], block_edges[1:])])
    detrended = power - np.interp(np.arange(len(power)), block_centres, block_medians)
    spread = robust_standard_deviation(detrended)
    sde_curve = detrended/spread if spread > 0 else detrended*0
    best = int(np.argmax(sde_curve))
    # Refinement around the peak
    period = float(periods[best])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fine_periods = np.linspace(period*(1 - 0.003), period*(1 + 0.003), 1500)
        fine_durations = np.unique(np.r_[durations, np.linspace(0.6*periodogram.duration[best], 1.4*periodogram.duration[best], 9)])
        fine_durations = fine_durations[fine_durations < 0.9*minimum_period]
        fine = model.power(fine_periods, fine_durations, objective="likelihood")
        refined = int(np.argmax(fine.power))
        period, duration, epoch = float(fine.period[refined]), float(fine.duration[refined]), float(fine.transit_time[refined])
        # With few transits, a multiple of the true period fits almost as well (the transits in between fall in gaps of the data, or
        # are not seen): the shortest period P/n that fits (nearly) as well as P is kept
        best_power = float(fine.power[refined])
        for divisor in range(2, 8):
            trial = period/divisor
            if trial < minimum_period:
                break
            test = model.power(np.linspace(trial*(1 - 0.002), trial*(1 + 0.002), 400), fine_durations, objective="likelihood")
            index = int(np.argmax(test.power))
            if test.power[index] >= 0.9*best_power:
                period, duration, epoch = float(test.period[index]), float(test.duration[index]), float(test.transit_time[index])
        statistics = model.compute_stats(period, duration, epoch)
    depth, depth_error = (float(value) for value in statistics["depth"])
    counts = np.asarray(statistics["per_transit_count"])
    # Red noise, and vetting depths measured on the central part of the transits (less sensitive than the box depths of the BLS to
    # transits observed only partially, e.g. ingress or egress only, which look shallower)
    beta = red_noise_factor(time, flux)
    phase_time = ((time - epoch + 0.5*period) % period) - 0.5*period
    cycle = np.round((time - epoch)/period).astype(int)
    out_of_transit = np.abs(phase_time) > 0.75*duration
    baseline = float(np.median(flux[out_of_transit])) if np.count_nonzero(out_of_transit) > 10 else 1.0
    point_scatter = robust_standard_deviation(flux[out_of_transit]) if np.count_nonzero(out_of_transit) > 10 else robust_standard_deviation(flux)
    central = np.abs(phase_time) < 0.3*duration
    depth_odd = window_depth(flux, central & (cycle % 2 == 1), baseline, point_scatter, beta)
    depth_even = window_depth(flux, central & (cycle % 2 == 0), baseline, point_scatter, beta)
    secondary_time = ((time - epoch) % period) - 0.5*period
    depth_secondary = window_depth(flux, np.abs(secondary_time) < 0.5*duration, baseline, point_scatter, beta)
    depth_error = depth_error*beta
    # Keep a light version of the periodogram (maximum in bins) for the plots
    bins = np.linspace(0, len(periods), 2001).astype(int)
    light_periods = np.array([periods[start:end][np.argmax(sde_curve[start:end])] for start, end in zip(bins[:-1], bins[1:]) if end > start])
    light_sde = np.array([np.max(sde_curve[start:end]) for start, end in zip(bins[:-1], bins[1:]) if end > start])
    return {"star_index": star_index, "period": period, "epoch": epoch, "duration": duration, "depth": depth, "depth_error": depth_error,
            "snr": depth/depth_error if depth_error > 0 else 0.0, "sde": float(sde_curve[best]),
            "depth_odd": depth_odd, "depth_even": depth_even, "depth_secondary": depth_secondary, "red_noise_factor": beta,
            "harmonic_delta_log_likelihood": float(statistics["harmonic_delta_log_likelihood"]),
            "number_of_transits": int(np.count_nonzero(counts >= minimum_points_per_transit)),
            "number_of_partial_transits": int(np.count_nonzero(counts > 0)),
            "periodogram_period": light_periods.astype(np.float32), "periodogram_sde": light_sde.astype(np.float32)}


# ======================================================================================================================================
# 6. Vetting
# ======================================================================================================================================

def flag_systematic_periods(stars, sde_threshold, minimum_shared_stars):
    """Periods (or their harmonics) found in many stars at once are caused by the observations (nights, systematics), not by planets."""
    strong = np.flatnonzero(stars["searched"] & (stars["sde"] >= 0.7*sde_threshold))
    periods = np.asarray(stars["period"])[strong]
    shared = np.zeros(len(stars), dtype=bool)
    for position, star in enumerate(strong):
        ratio = periods/periods[position]
        same = np.zeros(len(periods), dtype=bool)
        for harmonic in (0.5, 1.0, 2.0):
            same |= np.abs(ratio - harmonic) < 0.002*harmonic
        if np.count_nonzero(same) >= minimum_shared_stars:
            shared[star] = True
    return shared


def vet(stars, sde_threshold, snr_threshold, minimum_number_of_transits, maximum_depth_ppt, minimum_shared_stars):
    """Detection criteria and vetting flags; fills the columns 'detected', 'flags' and 'status'."""
    systematic = flag_systematic_periods(stars, sde_threshold, minimum_shared_stars)
    flags, status, detected = [], [], []
    for index, star in enumerate(stars):
        if not star["searched"]:
            flags.append("")
            status.append("not searched")
            detected.append(False)
            continue
        star_flags = []
        odd, odd_error = star["depth_odd_ppt"], star["depth_odd_error_ppt"]
        even, even_error = star["depth_even_ppt"], star["depth_even_error_ppt"]
        if np.isfinite(odd_error) and np.isfinite(even_error) and abs(odd - even) > 3*np.hypot(odd_error, even_error):
            star_flags.append("odd_even_depths")
        secondary, secondary_error = star["depth_secondary_ppt"], star["depth_secondary_error_ppt"]
        if np.isfinite(secondary_error) and secondary > 3*secondary_error and secondary > 0.1*star["depth_ppt"]:
            star_flags.append("secondary_eclipse")
        if star["harmonic_delta_log_likelihood"] > 0:
            star_flags.append("sinusoidal")
        if star["depth_ppt"] > maximum_depth_ppt:
            star_flags.append("too_deep")
        if systematic[index]:
            star_flags.append("shared_period")
        if star["variability"] > 2.5:
            star_flags.append("variable")
        is_detected = bool(star["sde"] >= sde_threshold and star["snr"] >= snr_threshold
                           and star["number_of_transits"] >= minimum_number_of_transits)
        detected.append(is_detected)
        flags.append(",".join(star_flags))
        if is_detected:
            status.append("candidate" if not star_flags else "likely false positive")
        else:
            status.append("")
    stars["detected"] = detected
    stars["flags"] = flags
    stars["status"] = status


# ======================================================================================================================================
# Result
# ======================================================================================================================================

class BlindSearchResult:

    """
    Result of photometry.blind_search().

    Attributes
    ----------
    stars: Table
        One row per catalogue star: id, ra, dec, magnitude (if target_magnitude was given), noise, BLS results (period in days, epoch, duration
        in hours, depth and errors in ppt, snr, sde, number_of_transits), vetting flags and status ("candidate", "likely false positive" or
        "" when nothing was detected).
    candidates: Table
        The detected stars (status "candidate" first), sorted by signal-to-noise ratio.
    frames: Table
        One row per frame: file name, instrument, night, time, airmass, FWHM, transparency, used.
    time: array
        Mid-exposure times of the frames (BJD_TDB, or MJD_UTC without site coordinates); time_scale gives the scale.
    relative_flux, relative_flux_error: arrays (stars x frames)
        Light curves after the removal of the systematics (1 = median brightness; NaN where the star was not measured or the frame was
        rejected).
    """

    def __init__(self, stars, frames, time, time_scale, relative_flux, relative_flux_error, periodograms, settings, reference_frames):
        self.stars = stars
        self.frames = frames
        self.time = np.asarray(time, dtype=float)
        self.time_scale = time_scale
        self.relative_flux = relative_flux
        self.relative_flux_error = relative_flux_error
        self.periodograms = periodograms
        self.settings = settings
        self.reference_frames = reference_frames

    @property
    def candidates(self):
        detected = self.stars[self.stars["detected"]]
        if len(detected) == 0:
            return detected
        order = np.lexsort((-np.asarray(detected["snr"]), np.asarray(detected["status"]) != "candidate"))
        return detected[order]

    def __repr__(self):
        number_of_candidates = int(np.count_nonzero(self.stars["status"] == "candidate"))
        number_of_false_positives = int(np.count_nonzero(self.stars["status"] == "likely false positive"))
        return (f"BlindSearchResult({len(self.stars)} stars, {int(np.count_nonzero(self.stars['searched']))} searched, {len(self.frames)} frames "
                f"on {len(np.unique(self.frames['night']))} nights; {number_of_candidates} candidates, {number_of_false_positives} likely false "
                f"positives)")

    def _row(self, star_id):
        rows = np.flatnonzero(self.stars["id"] == star_id)
        if len(rows) == 0:
            raise ValueError(f"No star with id {star_id}.")
        return int(rows[0])

    # ----------------------------------------------------------------------------------------------------------------------------------
    def light_curve(self, star_id, fold=False, window_hours=None):
        """
        Light curve of a star as a table (time, relative_flux, relative_flux_error, airmass, fwhm, instrument, night, used), in the format of
        photometry.differential_photometry(), so it can be given to photometry.fit_transit() and plot_light_curve(). Only the frames used
        are included.

        With fold=True, the times are folded with the BLS period around the transit closest to the middle of the data
        (time -> epoch + phase x period, that epoch is in meta["mid_transit_time"]): all transits fall on top of each other, so
        fit_transit() (which fixes the period) and its plots work with data of many nights. window_hours (with fold=True) keeps only the
        points closer than that to the mid-transit (faster fits, clearer plots).
        """
        row = self._row(star_id)
        good = np.isfinite(self.relative_flux[row]) & np.asarray(self.frames["used"])
        time = self.time[good]
        star = self.stars[row]
        middle_epoch = np.nan
        if np.isfinite(star["period"]):
            middle_epoch = star["epoch"] + np.round((np.median(time) - star["epoch"])/star["period"])*star["period"]
        if fold:
            if not np.isfinite(star["period"]):
                raise ValueError(f"Star {star_id} was not searched: no period to fold with.")
            time = middle_epoch + ((time - middle_epoch + 0.5*star["period"]) % star["period"]) - 0.5*star["period"]
        keep = np.ones(len(time), dtype=bool)
        if window_hours is not None:
            if not fold:
                raise ValueError("window_hours needs fold=True.")
            keep = np.abs(time - middle_epoch) < window_hours/24
        table = Table({"time": time, "relative_flux": self.relative_flux[row][good], "relative_flux_error": self.relative_flux_error[row][good],
                       "airmass": np.asarray(self.frames["airmass"])[good], "fwhm": np.asarray(self.frames["fwhm"])[good],
                       "instrument": np.asarray(self.frames["instrument"])[good], "night": np.asarray(self.frames["night"])[good],
                       "used": np.ones(np.count_nonzero(good), dtype=bool)})[keep]
        order = np.argsort(table["time"])
        table = table[order]
        table.meta.update({"time_scale": self.time_scale, "star_id": int(star_id), "ra": float(star["ra"]), "dec": float(star["dec"]),
                           "folded": bool(fold), "exposure_time": float(np.median(self.frames["exposure_time"]))})
        if np.isfinite(star["period"]):
            table.meta.update({"period": float(star["period"]), "epoch": float(star["epoch"]), "mid_transit_time": float(middle_epoch)})
        return table

    # ----------------------------------------------------------------------------------------------------------------------------------
    def save(self, output_directory):
        """Writes the star table, the candidates, the frame table (ECSV) and the light curves and periodograms (FITS)."""
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        files = []
        self.stars.meta.update({"time_scale": self.time_scale, "settings": self.settings,
                                "reference_frames": {key: str(value) for key, value in self.reference_frames.items()}})
        for name, table in (("blind_search_stars.ecsv", self.stars), ("blind_search_candidates.ecsv", self.candidates),
                            ("blind_search_frames.ecsv", self.frames)):
            table.write(output_directory/name, format="ascii.ecsv", overwrite=True)
            files.append(output_directory/name)
        periodogram_ids = sorted(self.periodograms)
        length = max([len(self.periodograms[star_id][0]) for star_id in periodogram_ids] or [1])
        periodogram_periods = np.full((len(periodogram_ids), length), np.nan, dtype=np.float32)
        periodogram_sde = np.full_like(periodogram_periods, np.nan)
        for index, star_id in enumerate(periodogram_ids):
            period, sde = self.periodograms[star_id]
            periodogram_periods[index, :len(period)], periodogram_sde[index, :len(sde)] = period, sde
        hdu_list = fits.HDUList([fits.PrimaryHDU(self.relative_flux.astype(np.float32)),
                                 fits.ImageHDU(self.relative_flux_error.astype(np.float32), name="ERROR"),
                                 fits.ImageHDU(self.time, name="TIME"),
                                 fits.ImageHDU(np.array(periodogram_ids, dtype=np.int32), name="PGRAM_ID"),
                                 fits.ImageHDU(periodogram_periods, name="PGRAM_P"), fits.ImageHDU(periodogram_sde, name="PGRAM_SDE")])
        hdu_list[0].header["TIMESCAL"] = self.time_scale
        hdu_list[0].header["COMMENT"] = "Relative flux (stars x frames) of the blind search; rows = stars of blind_search_stars.ecsv"
        hdu_list.writeto(output_directory/"blind_search_light_curves.fits", overwrite=True)
        files.append(output_directory/"blind_search_light_curves.fits")
        return files

    @classmethod
    def load(cls, output_directory):
        """Reads a result saved by save() (photometry.blind_search() saves it automatically in its output_directory)."""
        output_directory = Path(output_directory)
        stars = Table.read(output_directory/"blind_search_stars.ecsv", format="ascii.ecsv")
        frames = Table.read(output_directory/"blind_search_frames.ecsv", format="ascii.ecsv")
        with fits.open(output_directory/"blind_search_light_curves.fits") as hdu_list:
            relative_flux = hdu_list[0].data.astype(float)
            relative_flux_error = hdu_list["ERROR"].data.astype(float)
            time = hdu_list["TIME"].data.astype(float)
            ids = hdu_list["PGRAM_ID"].data
            periodograms = {}
            for index, star_id in enumerate(ids):
                period, sde = hdu_list["PGRAM_P"].data[index], hdu_list["PGRAM_SDE"].data[index]
                keep = np.isfinite(period)
                periodograms[int(star_id)] = (period[keep], sde[keep])
            time_scale = hdu_list[0].header.get("TIMESCAL", "BJD_TDB")
        return cls(stars, frames, time, time_scale, relative_flux, relative_flux_error, periodograms, dict(stars.meta.get("settings", {})),
                   dict(stars.meta.get("reference_frames", {})))

    # ----------------------------------------------------------------------------------------------------------------------------------
    def plot_summary(self, image_scale="log"):
        """Noise vs magnitude, signal-to-noise ratio vs period of all stars, and the field with the candidates (three figures)."""
        stars = self.stars
        searched = np.asarray(stars["searched"])
        magnitude_name = "magnitude" if np.any(np.isfinite(stars["magnitude"])) else "instrumental_magnitude"
        magnitude = np.asarray(stars[magnitude_name], dtype=float)
        status = np.asarray(stars["status"])
        styles = {"candidate": dict(color="#D62728", marker="o", s=70, label="candidate"),
                  "likely false positive": dict(color="#FF7F0E", marker="s", s=55, label="likely false positive")}
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            # 1. Noise
            figure, axis = plt.subplots(figsize=(9, 6))
            axis.scatter(magnitude[searched], stars["rms_ppt"][searched], s=12, color="0.45", label="stars (scatter per point)")
            order = np.argsort(magnitude[searched])
            axis.plot(magnitude[searched][order], np.asarray(stars["noise_trend_ppt"])[searched][order], color="k", lw=1.2,
                      label="typical scatter")
            expected = np.asarray(stars["expected_noise_ppt"])[searched][order]
            smoothed_expected = np.array([np.nanmedian(expected[max(index - 5, 0):index + 6]) for index in range(len(expected))])
            axis.plot(magnitude[searched][order], smoothed_expected, ":", color="C0", lw=1.5, label="expected (photon + sky + read noise)")
            for name, style in styles.items():
                chosen = status == name
                if np.any(chosen):
                    axis.scatter(magnitude[chosen], stars["rms_ppt"][chosen], zorder=3, edgecolor="k", **style)
                    for star in stars[chosen]:
                        axis.annotate(str(star["id"]), (star[magnitude_name], star["rms_ppt"]), textcoords="offset points", xytext=(6, 4),
                                      fontsize=9)
            target = np.asarray(stars["is_target"])
            if np.any(target):
                axis.scatter(magnitude[target], stars["rms_ppt"][target], marker="*", s=220, color="#17BECF", edgecolor="k", zorder=4,
                             label="target")
            axis.set_yscale("log")
            axis.set_xlabel("Magnitude" + (" (approximate, scaled to target_magnitude)" if magnitude_name == "magnitude" else
                                           " (instrumental)"))
            axis.set_ylabel("Scatter per point (ppt)")
            axis.set_title(f"Photometric precision: {np.count_nonzero(searched)} stars searched", fontsize=12)
            axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.5)
            axis.legend(fontsize=9)
            plt.tight_layout()
            plt.show()
            # 2. Signal-to-noise ratio vs period
            figure, axis = plt.subplots(figsize=(9, 6))
            passes_other_criteria = (np.asarray(stars["sde"]) >= self.settings.get("sde_threshold", 9.0)) & \
                (np.asarray(stars["number_of_transits"]) >= self.settings.get("minimum_number_of_transits", 2))
            for chosen, face, label in ((searched & passes_other_criteria, "0.45", "best period of each star"),
                                        (searched & ~passes_other_criteria, "none", "idem, below the SDE or number-of-transits criteria")):
                axis.scatter(stars["period"][chosen], stars["snr"][chosen], s=16, facecolor=face, edgecolor="0.45", label=label)
            for name, style in styles.items():
                chosen = status == name
                if np.any(chosen):
                    axis.scatter(stars["period"][chosen], stars["snr"][chosen], zorder=3, edgecolor="k", **style)
                    for star in stars[chosen]:
                        axis.annotate(str(star["id"]), (star["period"], star["snr"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
            axis.axhline(self.settings.get("snr_threshold", 8.0), color="k", ls="--", lw=1, label="SNR threshold")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Period (days)")
            axis.set_ylabel("Signal-to-noise ratio of the transit depth")
            axis.set_title("Box Least Squares: strongest signal of each star", fontsize=12)
            axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.5)
            axis.legend(fontsize=9)
            plt.tight_layout()
            plt.show()
            # 3. Field
            self.plot_field(image_scale=image_scale)

    def plot_field(self, instrument=None, image_scale="log"):
        """Reference frame of an instrument (default: the first one) with the searched stars, the candidates and the target."""
        from astropy.coordinates import SkyCoord
        from astropy.visualization import ImageNormalize, LinearStretch, LogStretch, PercentileInterval
        instrument = instrument or next(iter(self.reference_frames))
        source = self.reference_frames[instrument]
        image = _load(source)
        x, y = image.wcs.world_to_pixel(SkyCoord(self.stars["ra"], self.stars["dec"], unit="deg"))
        status = np.asarray(self.stars["status"])
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            figure, axis = plt.subplots(figsize=(9.5, 9))
            finite = image.data[np.isfinite(image.data)]
            norm = ImageNormalize(finite, interval=PercentileInterval(99.5), stretch=LogStretch(1000) if image_scale == "log" else LinearStretch())
            axis.imshow(image.data, origin="lower", cmap="magma", norm=norm, interpolation="nearest")
            searched = np.asarray(self.stars["searched"])
            axis.scatter(x[searched], y[searched], s=60, facecolor="none", edgecolor="0.8", lw=0.7, label="searched stars")
            for name, color in (("candidate", "#2ECC71"), ("likely false positive", "#FF7F0E")):
                chosen = status == name
                if np.any(chosen):
                    axis.scatter(x[chosen], y[chosen], s=260, facecolor="none", edgecolor=color, lw=2, label=name)
                    for star_id, x_star, y_star in zip(self.stars["id"][chosen], x[chosen], y[chosen]):
                        axis.annotate(str(star_id), (x_star, y_star), color=color, textcoords="offset points", xytext=(9, 9), fontsize=11,
                                      fontweight="bold")
            target = np.asarray(self.stars["is_target"])
            if np.any(target):
                axis.scatter(x[target], y[target], s=420, marker="o", facecolor="none", edgecolor="#4FC3F7", lw=2, ls="--", label="target")
            axis.set_xlim(-0.5, image.data.shape[1] - 0.5)
            axis.set_ylim(-0.5, image.data.shape[0] - 0.5)
            axis.set_title(f"{instrument}: {Path(str(source)).name}", fontsize=12)
            axis.set_xlabel("x (pixels)")
            axis.set_ylabel("y (pixels)")
            axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.4, color="white")
            axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
            plt.tight_layout()
            plt.show()

    def plot_candidate(self, star_id, number_of_points_per_bin=None, show_individual_transits=True):
        """
        Vetting sheet of a star: periodogram, folded light curve with the box model, odd and even transits, phase 0.5 (secondary eclipse),
        and (optionally) each observed transit night by night.
        """
        row = self._row(star_id)
        star = self.stars[row]
        if not star["searched"]:
            raise ValueError(f"Star {star_id} was not searched.")
        period, epoch, duration = star["period"], star["epoch"], star["duration_hours"]/24
        good = np.isfinite(self.relative_flux[row]) & np.asarray(self.frames["used"])
        time, flux = self.time[good], self.relative_flux[row][good]
        phase_time = ((time - epoch + 0.5*period) % period) - 0.5*period                      # days from mid-transit
        cycle = np.round((time - epoch)/period).astype(int)
        depth = star["depth_ppt"]/1e3
        window = max(3*duration, 0.1)
        bin_width = duration/6
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            figure, axes = plt.subplots(2, 2, figsize=(14, 10))
            # periodogram
            axis = axes[0, 0]
            if int(star["id"]) in self.periodograms:
                periodogram_period, periodogram_sde = self.periodograms[int(star["id"])]
                axis.plot(periodogram_period, periodogram_sde, color="0.3", lw=0.7)
            axis.axvline(period, color="#D62728", lw=1.2, alpha=0.6)
            for harmonic in (0.5, 2.0):
                axis.axvline(period*harmonic, color="#D62728", lw=0.8, ls=":", alpha=0.6)
            axis.set_xscale("log")
            axis.set_xlabel("Period (days)")
            axis.set_ylabel("SDE")
            axis.set_title(f"BLS periodogram: P = {period:.5f} d, SDE = {star['sde']:.1f}", fontsize=11)
            # folded light curve
            axis = axes[0, 1]
            near = np.abs(phase_time) < window
            hours = phase_time*24
            axis.plot(hours[near], flux[near], ".", color="0.6", ms=4, label="data")
            self._binned(axis, hours[near], flux[near], bin_width*24, label="binned")
            box_hours = np.array([-window, -duration/2, -duration/2, duration/2, duration/2, window])*24
            axis.plot(box_hours, 1 - depth*np.array([0, 0, 1, 1, 0, 0]), color="#D62728", lw=1.6, label="box model")
            axis.set_xlabel("Hours from mid-transit")
            axis.set_ylabel("Relative flux")
            axis.set_title(f"Folded: depth = {star['depth_ppt']:.2f} ± {star['depth_error_ppt']:.2f} ppt, duration = "
                           f"{star['duration_hours']:.2f} h, {star['number_of_transits']} transits", fontsize=11)
            axis.legend(fontsize=9)
            # odd and even
            axis = axes[1, 0]
            for parity, color, name, column in ((0, "#1F77B4", "even", "depth_even"), (1, "#FF7F0E", "odd", "depth_odd")):
                chosen = near & (np.abs(cycle) % 2 == parity)
                if np.any(chosen):
                    self._binned(axis, hours[chosen], flux[chosen], bin_width*24, color=color,
                                 label=f"{name} transits: {star[column + '_ppt']:.2f} ± {star[column + '_error_ppt']:.2f} ppt")
                    axis.axhline(1 - star[column + "_ppt"]/1e3, color=color, lw=1, ls="--")
            axis.set_xlabel("Hours from mid-transit")
            axis.set_ylabel("Relative flux (binned)")
            axis.set_title("Odd and even transits (different depths: eclipsing binary at twice the period)", fontsize=11)
            axis.legend(fontsize=9)
            # secondary
            axis = axes[1, 1]
            secondary_time = ((time - epoch) % period) - 0.5*period
            chosen = np.abs(secondary_time) < window
            axis.plot(secondary_time[chosen]*24, flux[chosen], ".", color="0.6", ms=4)
            self._binned(axis, secondary_time[chosen]*24, flux[chosen], bin_width*24, label="binned")
            axis.axhline(1 - star["depth_secondary_ppt"]/1e3, color="#9467BD", lw=1.2, ls="--",
                         label=f"depth at phase 0.5: {star['depth_secondary_ppt']:.2f} ± {star['depth_secondary_error_ppt']:.2f} ppt")
            axis.axhline(1, color="k", lw=0.8)
            axis.set_xlabel("Hours from phase 0.5")
            axis.set_ylabel("Relative flux")
            axis.set_title("Phase 0.5 (secondary eclipse: eclipsing binary)", fontsize=11)
            axis.legend(fontsize=9)
            for axis in axes.ravel():
                axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.5)
                axis.tick_params(direction="in", top=True, right=True)
            magnitude = star["magnitude"] if np.isfinite(star["magnitude"]) else star["instrumental_magnitude"]
            flags = star["flags"] or "none"
            figure.suptitle(f"Star {star['id']} (RA {star['ra']:.5f}, Dec {star['dec']:+.5f}, mag {magnitude:.2f}): "
                            f"{star['status'] or 'not detected'};  SNR = {star['snr']:.1f};  flags: {flags}", fontsize=13)
            plt.tight_layout()
            plt.show()
            if show_individual_transits:
                events = np.unique(cycle[near & (np.abs(phase_time) < duration/2)])
                if len(events):
                    columns = min(len(events), 4)
                    rows = int(np.ceil(len(events)/columns))
                    figure, axes = plt.subplots(rows, columns, figsize=(3.6*columns, 3.0*rows), squeeze=False, sharey=True)
                    nights = np.asarray(self.frames["night"])[good]
                    instruments = np.asarray(self.frames["instrument"])[good]
                    for axis, event in zip(axes.ravel(), events):
                        chosen = (cycle == event) & near
                        axis.plot(hours[chosen], flux[chosen], ".", color="0.45", ms=4)
                        axis.plot(box_hours, 1 - depth*np.array([0, 0, 1, 1, 0, 0]), color="#D62728", lw=1.2)
                        axis.set_xlim(-window*24, window*24)
                        axis.set_title(f"{nights[chosen][0]} ({instruments[chosen][0].split(' / ')[0]})", fontsize=9)
                        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
                    for axis in axes.ravel()[len(events):]:
                        axis.axis("off")
                    figure.supxlabel("Hours from mid-transit")
                    figure.supylabel("Relative flux")
                    figure.suptitle(f"Star {star['id']}: each observed transit", fontsize=12)
                    plt.tight_layout()
                    plt.show()

    @staticmethod
    def _binned(axis, x, y, width, color="k", label=None):
        if len(x) == 0:
            return
        edges = np.arange(x.min(), x.max() + width, width)
        index = np.digitize(x, edges)
        centres, means, errors = [], [], []
        for value in np.unique(index):
            chosen = index == value
            if np.count_nonzero(chosen) >= 3:
                centres.append(np.mean(x[chosen]))
                means.append(np.mean(y[chosen]))
                errors.append(np.std(y[chosen])/np.sqrt(np.count_nonzero(chosen)))
        axis.errorbar(centres, means, errors, fmt="o", ms=5, color=color, capsize=2, zorder=3, label=label)


# ======================================================================================================================================
# Main function
# ======================================================================================================================================

def run_blind_search(images, target_coordinates=None, target_magnitude=None, minimum_period=0.5, maximum_period=10.0,
                     transit_durations_hours=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0), sde_threshold=9.0, snr_threshold=8.0,
                     minimum_number_of_transits=2, maximum_depth=0.05, number_of_systematics=3, maximum_number_of_stars=300,
                     detection_threshold=5.0, minimum_number_of_points=100, minimum_transparency=0.4, maximum_fwhm_factor=2.0,
                     frequency_factor=1.0, aperture_radii=DEFAULT_APERTURE_RADII, sky_annulus=DEFAULT_SKY_ANNULUS, number_of_cores=None,
                     output_directory="blind_search_output", reuse_measurements=True, verbose=True):
    """See photometry.blind_search() for the parameters. Returns a BlindSearchResult."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    cores = number_of_workers(number_of_cores)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if isinstance(images, aux.Image) or (isinstance(images, (list, tuple)) and images and isinstance(images[0], aux.Image)):
        sources = [images] if isinstance(images, aux.Image) else list(images)
    else:
        sources = collect_frames(images)
    if target_coordinates is not None and not isinstance(target_coordinates, SkyCoord):
        target_coordinates = SkyCoord(*target_coordinates, unit="deg")
    # 1. Inventory
    if verbose:
        print(f"1. Inventory of {len(sources)} frames ({cores} processes)")
    frames = [_frame_information(source) for source in sources]
    missing_wcs = [frame["name"] for frame in frames if not frame["has_wcs"]]
    if missing_wcs:
        raise ValueError(f"{len(missing_wcs)} frames have no WCS (e.g. {missing_wcs[0]}): plate-solve them (e.g. with astrometry.net, "
                         "ASTAP or the acquisition software) before the blind search.")
    instruments = np.array([frame["instrument"] for frame in frames])
    sources_by_instrument = {instrument: [source for source, owner in zip(sources, instruments) if owner == instrument]
                             for instrument in dict.fromkeys(instruments)}
    # 2. Catalogue
    if verbose:
        print(f"2. Star catalogue from the best frame of each of the {len(sources_by_instrument)} instruments")
    catalogue, reference_frames, fwhm_by_instrument = build_catalogue(sources_by_instrument, detection_threshold, maximum_number_of_stars,
                                                                       cores, verbose)
    coordinates = SkyCoord(catalogue["ra"], catalogue["dec"], unit="deg")
    field_center = target_coordinates or SkyCoord(np.median(catalogue["ra"]), np.median(catalogue["dec"]), unit="deg")
    times, time_scale, nights, middle_mjd = mid_exposure_times(frames, field_center)
    nights = np.array(nights)
    # 3. Photometry
    if verbose:
        print(f"3. Aperture photometry of {len(catalogue)} stars in {len(frames)} frames")
    cache_directory = output_directory/"measurements"
    cache_directory.mkdir(exist_ok=True)
    key = catalogue_hash(catalogue, aperture_radii, sky_annulus)
    tasks = []
    for source, frame, instrument in zip(sources, frames, instruments):
        cache_file = None
        if reuse_measurements and not isinstance(source, aux.Image):
            # the file is identified by its full path and modification time (reduced frames of different nights can have the same name)
            identity = f"{Path(source).resolve()}|{Path(source).stat().st_mtime_ns}"
            cache_file = cache_directory/f"{Path(source).stem}_{hashlib.sha1(identity.encode()).hexdigest()[:12]}_{key}.npz"
        tasks.append((source, np.asarray(catalogue["ra"]), np.asarray(catalogue["dec"]), fwhm_by_instrument[instrument], aperture_radii,
                      sky_annulus, cache_file))
    cached = sum(1 for task in tasks if task[-1] is not None and Path(task[-1]).exists())
    if verbose and cached:
        print(f"   {cached} frames already measured (cache in {cache_directory})")
    measurements = parallel_map(_measure_frame, tasks, cores, "frames measured", verbose)
    # 4. Light curves
    if verbose:
        print("4. Light curves: normalization per instrument, frame zero points, best aperture, SysRem")
    light_curves = build_light_curves(measurements, frames, instruments, nights, number_of_systematics, minimum_transparency,
                                      maximum_fwhm_factor, verbose)
    residuals, residual_errors = light_curves["residuals"], light_curves["errors"]
    relative_flux = 10**(-0.4*residuals)
    noise_scale = np.clip(light_curves["point_to_point"]/np.maximum(light_curves["expected"], 1e-6), 1, None)
    relative_flux_error = MAGNITUDE_TO_RELATIVE_FLUX*residual_errors*relative_flux*noise_scale[:, None]
    airmass = airmass_of(frames, middle_mjd, field_center)
    frame_table = Table({"file": [str(source) if not isinstance(source, aux.Image) else frame["name"] for source, frame in zip(sources, frames)], "instrument": instruments, "night": nights, "time": times,
                         "exposure_time": [float(frame["header_items"].get("EXPTIME", np.nan)) for frame in frames], "airmass": airmass,
                         "fwhm": light_curves["fwhm"], "transparency": light_curves["transparency"], "used": light_curves["used"]})
    # Star table
    stars = catalogue.copy()
    stars["magnitude"] = np.full(len(stars), np.nan)
    target_row = None
    if target_coordinates is not None:
        separation = target_coordinates.separation(coordinates).arcsec
        stars["distance_to_target_arcsec"] = separation
        target_row = int(np.argmin(separation))
        if separation[target_row] > 5:
            warnings.warn(f"No catalogue star within 5\" of the target coordinates (closest: {separation[target_row]:.1f}\").")
            target_row = None
    stars["is_target"] = np.zeros(len(stars), dtype=bool)
    if target_row is not None:
        stars["is_target"][target_row] = True
        if target_magnitude is not None:
            stars["magnitude"] = stars["instrumental_magnitude"] - stars["instrumental_magnitude"][target_row] + target_magnitude
    used = light_curves["used"]
    number_of_points = np.sum(np.isfinite(residuals[:, used]), axis=1)
    number_of_nights = np.array([len(np.unique(nights[used][np.isfinite(residuals[index, used])])) for index in range(len(stars))])
    stars["aperture_radius_fwhm"] = np.asarray(aperture_radii)[light_curves["best_radius"]]
    stars["number_of_points"] = number_of_points
    stars["number_of_nights"] = number_of_nights
    stars["saturated_fraction"] = light_curves["saturated_fraction"]
    stars["rms_ppt"] = 1e3*MAGNITUDE_TO_RELATIVE_FLUX*light_curves["scatter"]
    stars["point_to_point_ppt"] = 1e3*MAGNITUDE_TO_RELATIVE_FLUX*light_curves["point_to_point"]
    stars["expected_noise_ppt"] = 1e3*MAGNITUDE_TO_RELATIVE_FLUX*light_curves["expected"]
    searched = (number_of_points >= minimum_number_of_points) & (number_of_nights >= 2) & (stars["saturated_fraction"] < 0.05)
    stars["searched"] = searched
    # typical scatter as a function of brightness (running median), to find variable stars
    order = np.argsort(stars["instrumental_magnitude"])
    trend = np.full(len(stars), np.nan)
    searched_order = order[searched[order]]
    for position, index in enumerate(searched_order):
        neighbours = searched_order[max(position - 7, 0):position + 8]
        trend[index] = np.nanmedian(stars["point_to_point_ppt"][neighbours])
    stars["noise_trend_ppt"] = trend
    # 5. BLS
    baseline = np.nanmax(times[used]) - np.nanmin(times[used])
    longest_period = min(maximum_period, 0.5*baseline)
    durations = np.array(transit_durations_hours)/24
    durations = durations[durations < 0.9*minimum_period]
    if verbose:
        print(f"5. Box Least Squares search of {np.count_nonzero(searched)} stars: periods {minimum_period}-{longest_period:.1f} d, "
              f"durations {transit_durations_hours[0]}-{transit_durations_hours[-1]} h")
    tasks = []
    for index in np.flatnonzero(searched):
        good = np.isfinite(relative_flux[index]) & used
        tasks.append((index, times[good], relative_flux[index][good], relative_flux_error[index][good], minimum_period, longest_period,
                      durations, frequency_factor, 4))
    searches = parallel_map(_search_star, tasks, cores, "stars searched", verbose)
    columns = {name: np.full(len(stars), np.nan) for name in ("period", "epoch", "duration_hours", "depth_ppt", "depth_error_ppt", "snr", "sde",
                                                               "depth_odd_ppt", "depth_odd_error_ppt", "depth_even_ppt", "depth_even_error_ppt",
                                                               "depth_secondary_ppt", "depth_secondary_error_ppt",
                                                               "harmonic_delta_log_likelihood", "red_noise_factor")}
    number_of_transits = np.zeros(len(stars), dtype=int)
    periodograms = {}
    for search in searches:
        index = search["star_index"]
        columns["period"][index], columns["epoch"][index] = search["period"], search["epoch"]
        columns["duration_hours"][index] = search["duration"]*24
        columns["depth_ppt"][index], columns["depth_error_ppt"][index] = 1e3*search["depth"], 1e3*search["depth_error"]
        columns["snr"][index], columns["sde"][index] = search["snr"], search["sde"]
        for name in ("depth_odd", "depth_even", "depth_secondary"):
            columns[name + "_ppt"][index], columns[name + "_error_ppt"][index] = 1e3*search[name][0], 1e3*search[name][1]
        columns["harmonic_delta_log_likelihood"][index] = search["harmonic_delta_log_likelihood"]
        columns["red_noise_factor"][index] = search["red_noise_factor"]
        number_of_transits[index] = search["number_of_transits"]
        periodograms[int(stars["id"][index])] = (search["periodogram_period"], search["periodogram_sde"])
    for name, values in columns.items():
        stars[name] = values
    stars["number_of_transits"] = number_of_transits
    # variability out of transit (relative to stars of the same brightness)
    variability = np.full(len(stars), np.nan)
    for index in np.flatnonzero(searched):
        phase = ((times - stars["epoch"][index] + 0.5*stars["period"][index]) % stars["period"][index]) - 0.5*stars["period"][index]
        outside = used & np.isfinite(residuals[index]) & (np.abs(phase) > stars["duration_hours"][index]/24)
        if np.count_nonzero(outside) > 20:
            variability[index] = (1e3*MAGNITUDE_TO_RELATIVE_FLUX*point_to_point_scatter_by_night(residuals[index:index + 1, outside],
                                                                                                 nights[outside])[0]
                                  / max(stars["noise_trend_ppt"][index], 1e-3))
            variability[index] = max(variability[index], 1e3*MAGNITUDE_TO_RELATIVE_FLUX*robust_standard_deviation(residuals[index, outside])
                                     / max(stars["noise_trend_ppt"][index], 1e-3))
    stars["variability"] = variability
    vet(stars, sde_threshold, snr_threshold, minimum_number_of_transits, 1e3*maximum_depth,
        minimum_shared_stars=max(3, int(0.03*np.count_nonzero(searched))))
    for name in ("ra", "dec"):
        stars[name].unit = u.deg
    settings = {"minimum_period": minimum_period, "maximum_period": float(longest_period), "transit_durations_hours": list(transit_durations_hours),
                "sde_threshold": sde_threshold, "snr_threshold": snr_threshold, "minimum_number_of_transits": minimum_number_of_transits,
                "maximum_depth": maximum_depth, "number_of_systematics": number_of_systematics,
                "aperture_radii_fwhm": list(aperture_radii), "sky_annulus_fwhm": list(sky_annulus)}
    result = BlindSearchResult(stars, frame_table, times, time_scale, relative_flux, relative_flux_error, periodograms, settings,
                               reference_frames)
    files = result.save(output_directory)
    if verbose:
        number_of_candidates = int(np.count_nonzero(stars["status"] == "candidate"))
        number_of_false_positives = int(np.count_nonzero(stars["status"] == "likely false positive"))
        print(f"6. Vetting: {number_of_candidates} candidates and {number_of_false_positives} likely false positives "
              f"(criteria: SDE >= {sde_threshold}, SNR >= {snr_threshold}, at least {minimum_number_of_transits} transits)")
        for star in result.candidates:
            print(f"   star {star['id']:4d}: P = {star['period']:.5f} d, depth = {star['depth_ppt']:.2f} ± {star['depth_error_ppt']:.2f} ppt, "
                  f"duration = {star['duration_hours']:.2f} h, SNR = {star['snr']:.1f}, SDE = {star['sde']:.1f}, "
                  f"{star['number_of_transits']} transits -> {star['status']}" + (f" ({star['flags']})" if star["flags"] else ""))
        print(f"   results saved in {output_directory}: " + ", ".join(Path(file).name for file in files))
    return result
