"""
Auxiliary functions of easyspec.photometry (common tools and aperture photometry): reading reduced images, background, star detection,
FWHM, aperture photometry with aperture correction, cross-matching of positions and catalogues, and the photometric calibration with
standard stars.

The heavy lifting (backgrounds, star finding, apertures) is done by photutils; this module adds the robust choices, the error propagation
and the bookkeeping. It works with photutils 2.x and 3.x.
"""

import inspect
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import QTable, Table
from astropy.wcs import WCS
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D, MADStdBackgroundRMS, MedianBackground
from photutils.detection import DAOStarFinder

from easyspec import data_io

MAGNITUDE_ERROR_FACTOR = 2.5/np.log(10)     # sigma_mag = 1.0857 sigma_flux/flux
GAUSSIAN_FWHM_TO_SIGMA = 1/(2*np.sqrt(2*np.log(2)))

# Bits of the "flags" column of the photometry tables:
FLAG_BITS = {1: "saturated (peak above the saturation level)",
             2: "bad pixels in the aperture (data quality mask: non-finite or flat field not applied)",
             4: "cosmic-ray pixels (replaced by LACosmic) in the aperture",
             8: "aperture or sky annulus partly outside the image",
             16: "crowded: another detected source closer than (aperture radius + 1.5 FWHM) and brighter than 1% of this star",
             32: "low signal-to-noise (S/N < 5)",
             64: "PSF fit problem (fit did not converge, position at the edge of its bounds, or poor fit quality)"}


# ======================================================================================================================================
# Images
# ======================================================================================================================================

@dataclass
class Image:
    """A reduced image (in electrons) with its variance, data quality mask and the metadata needed for photometry."""
    name: str
    data: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    header: fits.Header
    band: str = None
    exposure_time: float = None
    airmass: float = None
    mjd: float = None
    object_name: str = None
    saturation_level: float = np.inf
    wcs: object = None
    background: np.ndarray = None
    background_rms: np.ndarray = None
    fwhm: float = None
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Image({self.name}: {self.data.shape[1]}x{self.data.shape[0]} px, band={self.band}, exposure_time={self.exposure_time} s, "
                f"airmass={self.airmass}, object={self.object_name})")

    @property
    def bad_pixels(self):
        """Boolean image of unusable pixels (non-finite values, mask bit 4)."""
        return (self.mask & 4).astype(bool) | ~np.isfinite(self.data)


def header_value(header, keywords, default=None, cast=float):
    """Value of the first keyword of 'keywords' present in the header, converted with 'cast' (default if none is found or valid)."""
    for keyword in np.atleast_1d(keywords):
        if keyword in header:
            try:
                return cast(header[keyword])
            except (TypeError, ValueError):
                continue
    return default


def read_image(path, band_keyword="FILTER", exposure_time_keyword="EXPTIME", airmass_keyword="AIRMASS", saturation_keyword="SATURATE",
               gain_keyword="GAIN"):

    """
    Read an image reduced with easyspec.cleaning (data in electrons, with VAR and MASK extensions). Older files without VAR get a
    Poisson + read-noise variance (see easyspec.data_io.read_reduced_frame).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data, variance, mask, header = data_io.read_reduced_frame(path)
    if variance is None:
        variance = np.clip(data, 0, None) + 1.0
    in_electrons = str(header.get("BUNIT", "")).strip().lower() in ("electron", "electrons", "e-")
    if not in_electrons:
        warnings.warn(f"{Path(path).name} is not in electrons (BUNIT = '{header.get('BUNIT', '')}'): run cleaning.gain_correct() before "
                      "saving the reduced images, otherwise the uncertainties are wrong.")
    gain = header_value(header, gain_keyword, 1.0)
    saturation_in_adu = header_value(header, saturation_keyword, None)
    # Saturation in electrons (85% of the ADC limit, to stay away from the non-linear regime; the bias level is unknown here):
    saturation_level = 0.85*saturation_in_adu*(gain if in_electrons else 1.0) if saturation_in_adu else np.inf
    if str(header.get("COMBTYPE", "")).strip().lower() == "median" and header_value(header, "NCOMBINE", 1) > 1:
        warnings.warn(f"{Path(path).name} is a median of {int(header['NCOMBINE'])} frames. The pixel-by-pixel median of frames with "
                      "different seeing does not conserve the flux of the stars (biases of ~0.5-1% were measured in tests): for photometry "
                      "combine the aligned frames with cleaning.master('target', ..., method='mean') (the cosmic rays were already removed "
                      "by remove_cosmic_rays()).")
    wcs = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate_wcs = WCS(header)
        # A WCS without reference pixel (cleaning.master() drops CRPIX when it differs between the combined frames) is not valid
        if candidate_wcs.has_celestial and all(keyword in header for keyword in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2")):
            wcs = candidate_wcs
    except Exception:
        wcs = None
    mjd = header_value(header, ("MJD-OBS", "MJD"), None)
    if mjd is None and "DATE-OBS" in header:
        try:
            from astropy.time import Time
            mjd = Time(header["DATE-OBS"]).mjd
        except Exception:
            mjd = None
    return Image(name=Path(path).name, data=data, variance=variance, mask=mask, header=header,
                 band=str(header.get(band_keyword, "")).strip() or None, exposure_time=header_value(header, exposure_time_keyword, None),
                 airmass=header_value(header, airmass_keyword, None), mjd=mjd, object_name=str(header.get("OBJECT", "")).strip() or None,
                 saturation_level=saturation_level, wcs=wcs)


def collect_image_paths(source, extensions=("fits", "fit", "fts")):
    """Sorted list of image paths from a directory, a single file, or a list of files and/or directories."""
    if isinstance(source, (str, Path)):
        source = [source]
    paths = []
    for item in source:
        item = Path(item)
        if item.is_dir():
            paths += sorted(path for path in item.iterdir() if path.suffix.lower().lstrip(".") in extensions)
        elif item.exists():
            paths.append(item)
        else:
            raise FileNotFoundError(f"{item} does not exist.")
    if not paths:
        raise FileNotFoundError(f"No images found in {source}.")
    return paths


def pixel_scale_of(image):
    """Pixel scale (arcsec/pixel) from the WCS or from the PIXSCALE keyword (NaN if unknown)."""
    if image.wcs is not None:
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            return float(np.mean(proj_plane_pixel_scales(image.wcs.celestial))*3600)
        except Exception:
            pass
    return header_value(image.header, ("PIXSCALE", "SECPIX", "PIXSCAL1"), np.nan)


def parse_section(section):
    """FITS section string '[x1:x2,y1:y2]' (1-based, inclusive) -> python slices (y_slice, x_slice)."""
    match = re.fullmatch(r"\s*\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]\s*", str(section))
    if match is None:
        raise ValueError(f"Invalid FITS section '{section}'.")
    x_start, x_end, y_start, y_end = (int(value) for value in match.groups())
    return slice(y_start - 1, y_end), slice(x_start - 1, x_end)


# ======================================================================================================================================
# Background and sources
# ======================================================================================================================================

def estimate_background(image, box_size=64, filter_size=3, source_mask=None):
    """2D background and background RMS (photutils Background2D, sigma-clipped medians in boxes), ignoring bad and source pixels."""
    bad_pixels = image.bad_pixels
    mask = bad_pixels if source_mask is None else (bad_pixels | source_mask)
    box_size = int(min(box_size, max(16, min(image.data.shape)//4)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The keyword of the RMS estimator changed in photutils 3.0 (bkgrms_estimator -> bkg_rms_estimator)
        rms_keyword = "bkg_rms_estimator" if "bkg_rms_estimator" in inspect.signature(Background2D).parameters else "bkgrms_estimator"
        background_model = Background2D(np.where(bad_pixels, 0.0, image.data), box_size, mask=mask, filter_size=filter_size,
                                        sigma_clip=SigmaClip(sigma=3.0, maxiters=10), bkg_estimator=MedianBackground(),
                                        exclude_percentile=50.0, **{rms_keyword: MADStdBackgroundRMS()})
    return background_model.background, background_model.background_rms


def source_mask_from_threshold(data, background_level, background_rms, number_of_sigmas=2.0, dilation_iterations=5):
    """Pixels belonging to sources (above number_of_sigmas x RMS), dilated: used to compute a cleaner background in a second pass."""
    mask = (data - background_level) > number_of_sigmas*background_rms
    mask = ndimage.binary_opening(mask, iterations=1)
    return ndimage.binary_dilation(mask, iterations=dilation_iterations)


def centroid_column_names(table):
    """Names of the centroid columns of a photutils finder table (they changed in photutils 3.0)."""
    for x_name, y_name in (("x_centroid", "y_centroid"), ("xcentroid", "ycentroid")):
        if x_name in table.colnames:
            return x_name, y_name
    raise KeyError(f"No centroid columns in {table.colnames}.")


def detect_stars(image, threshold_in_sigmas=5.0, fwhm=3.0, maximum_number_of_sources=None, border=5, minimum_separation=2.5,
                 data=None, threshold_image=None):

    """
    Point sources detected with DAOStarFinder on the background-subtracted image (detections closer than minimum_separation FWHM are
    merged). Returns a QTable with x, y (0-based pixels), peak and a rough flux, sorted by decreasing peak.

    'data' replaces the background-subtracted image (e.g. a residual image after PSF subtraction), and 'threshold_image' replaces the
    background RMS (e.g. a noise map that includes the PSF-model errors); the threshold is then threshold_in_sigmas x threshold_image.
    """

    residual = image.data - image.background if data is None else data
    bad_pixels = image.bad_pixels | ~np.isfinite(residual)
    residual = np.where(bad_pixels, 0.0, residual)
    if threshold_image is not None:
        # DAOStarFinder takes a scalar threshold: detect on the S/N image instead
        residual = residual/np.where(threshold_image > 0, threshold_image, np.inf)
        threshold = threshold_in_sigmas
    else:
        threshold = threshold_in_sigmas*float(np.median(image.background_rms))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # minimum_separation (in FWHM) is explicit because its default changed in photutils 3.0 (from 0, which gives spurious detections
        # on saturated stars and on the wings of bright stars, to 2.5 FWHM)
        finder = DAOStarFinder(threshold=threshold, fwhm=float(fwhm), exclude_border=True, min_separation=minimum_separation*float(fwhm))
        found = finder(residual, mask=bad_pixels)
    if found is None or len(found) == 0:
        return QTable(names=("id", "x", "y", "peak", "flux_estimate"))
    x_name, y_name = centroid_column_names(found)
    table = QTable()
    table["x"] = np.asarray(found[x_name], dtype=float)
    table["y"] = np.asarray(found[y_name], dtype=float)
    table["peak"] = np.asarray(found["peak"], dtype=float)
    table["flux_estimate"] = np.asarray(found["flux"], dtype=float)
    if threshold_image is not None:            # back to electrons
        row_index = np.clip(np.round(table["y"]).astype(int), 0, image.data.shape[0] - 1)
        column_index = np.clip(np.round(table["x"]).astype(int), 0, image.data.shape[1] - 1)
        table["peak"] = table["peak"]*threshold_image[row_index, column_index]
        table["flux_estimate"] = table["flux_estimate"]*threshold_image[row_index, column_index]
    number_of_rows, number_of_columns = image.data.shape
    inside = ((table["x"] > border) & (table["x"] < number_of_columns - 1 - border) & (table["y"] > border)
              & (table["y"] < number_of_rows - 1 - border))
    table = table[inside]
    table.sort("peak", reverse=True)
    if maximum_number_of_sources is not None:
        table = table[:maximum_number_of_sources]
    table.add_column(np.arange(1, len(table) + 1), name="id", index=0)
    return table


def select_isolated_stars(image, sources, isolation_radius, maximum_neighbour_ratio=0.05, margin=0, minimum_peak_in_sigmas=0.0):
    """
    Indices (sorted by decreasing peak) of unsaturated sources with no neighbour brighter than maximum_neighbour_ratio x their peak within
    isolation_radius, at least 'margin' pixels from the image edges, and with a peak above minimum_peak_in_sigmas x background RMS.
    """
    if len(sources) == 0:
        return []
    positions = np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)]
    peaks = np.asarray(sources["peak"], dtype=float)
    tree = cKDTree(positions)
    number_of_rows, number_of_columns = image.data.shape
    background_level = float(np.median(image.background)) if image.background is not None else 0.0
    background_rms = float(np.median(image.background_rms)) if image.background_rms is not None else 0.0
    selected = []
    for index in np.argsort(-peaks):
        if peaks[index] + background_level > image.saturation_level or peaks[index] < minimum_peak_in_sigmas*background_rms:
            continue
        x, y = positions[index]
        if not (margin <= x < number_of_columns - margin - 1 and margin <= y < number_of_rows - margin - 1):
            continue
        neighbours = tree.query_ball_point(positions[index], isolation_radius)
        if any(other != index and peaks[other] > maximum_neighbour_ratio*peaks[index] for other in neighbours):
            continue
        selected.append(int(index))
    return selected


def measure_fwhm(image, sources, number_of_stars=40, stamp_size=None, isolation_radius=None):
    """
    Median FWHM (pixels) of bright, unsaturated and isolated stars, from 2D elliptical Gaussian fits to the stars' cores (a Gaussian fit to a
    Moffat or seeing profile recovers its FWHM within a few percent). Returns (fwhm, scatter, number of stars used).
    """
    if len(sources) == 0:
        return np.nan, np.nan, 0
    first_guess = 3.0
    stamp_size = stamp_size or 21
    half_size = stamp_size//2
    isolation_radius = isolation_radius or 2.5*stamp_size/2
    candidates = select_isolated_stars(image, sources, isolation_radius, margin=half_size)[:number_of_stars]
    positions = np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)]
    widths = []
    y_grid, x_grid = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    for index in candidates:
        x_center, y_center = int(round(positions[index, 0])), int(round(positions[index, 1]))
        window = (slice(y_center - half_size, y_center + half_size + 1), slice(x_center - half_size, x_center + half_size + 1))
        stamp = image.data[window] - image.background[window]
        if not np.all(np.isfinite(stamp)) or stamp.max() <= 0:
            continue
        x_offset, y_offset = positions[index, 0] - x_center, positions[index, 1] - y_center

        def residual(parameters):
            amplitude, x_mean, y_mean, sigma_major, sigma_minor, angle = parameters
            cosine, sine = np.cos(angle), np.sin(angle)
            u = (x_grid - x_mean)*cosine + (y_grid - y_mean)*sine
            v = -(x_grid - x_mean)*sine + (y_grid - y_mean)*cosine
            return (amplitude*np.exp(-0.5*((u/sigma_major)**2 + (v/sigma_minor)**2)) - stamp).ravel()

        sigma_guess = first_guess*GAUSSIAN_FWHM_TO_SIGMA
        try:
            fit = least_squares(residual, [stamp.max(), x_offset, y_offset, sigma_guess, sigma_guess, 0.0],
                                bounds=([0, -3, -3, 0.3, 0.3, -np.pi], [np.inf, 3, 3, half_size, half_size, np.pi]))
        except Exception:
            continue
        if fit.success:
            widths.append(np.sqrt(abs(fit.x[3]*fit.x[4]))/GAUSSIAN_FWHM_TO_SIGMA)
    if not widths:
        return np.nan, np.nan, 0
    widths = np.array(widths)
    median_width = float(np.median(widths))
    return median_width, float(1.4826*np.median(np.abs(widths - median_width))), len(widths)


# ======================================================================================================================================
# Aperture photometry
# ======================================================================================================================================

def local_sky_in_annuli(image, positions, inner_radius, outer_radius, excluded_pixels=None):
    """
    Local sky per source: sigma-clipped median and standard deviation in the annulus (masked pixels and pixels of other sources excluded),
    and the number of pixels used. 'excluded_pixels' is a boolean image of pixels to ignore (e.g. other stars).
    """
    annulus = CircularAnnulus(positions, inner_radius, outer_radius)
    unusable = image.bad_pixels
    if excluded_pixels is not None:
        unusable = unusable | excluded_pixels
    unusable = unusable.astype(float)
    sky_level = np.full(len(positions), np.nan)
    sky_standard_deviation = np.full(len(positions), np.nan)
    number_of_sky_pixels = np.zeros(len(positions), int)
    for index, annulus_mask in enumerate(annulus.to_mask(method="center")):
        values = annulus_mask.get_values(image.data)
        excluded = annulus_mask.get_values(unusable) > 0
        values = values[~excluded & np.isfinite(values)]
        if len(values) < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, median, standard_deviation = sigma_clipped_stats(values, sigma=3.0, maxiters=10)
        # The mode estimator (2.5 median - 1.5 mean, as in DAOPHOT/SExtractor) is biased in crowded annuli; the clipped median is more robust
        sky_level[index], sky_standard_deviation[index], number_of_sky_pixels[index] = median, standard_deviation, len(values)
    return sky_level, sky_standard_deviation, number_of_sky_pixels


def aperture_fluxes(image, positions, radius, sky_level, sky_standard_deviation, number_of_sky_pixels):
    """
    Sky-subtracted fluxes in circular apertures and their uncertainties: variance of the pixels (from the VAR extension: photon, read and
    calibration noise) plus the uncertainty of the sky level, (N_aperture sigma_sky)^2/N_sky. Returns flux, error, aperture area.
    """
    aperture = CircularAperture(positions, radius)
    bad_pixels = image.bad_pixels
    data = np.where(bad_pixels, 0.0, image.data)
    variance = np.where(bad_pixels | ~np.isfinite(image.variance), 0.0, np.clip(image.variance, 0, None))
    sums = np.asarray(aperture_photometry(data, aperture, method="exact")["aperture_sum"], dtype=float)
    variance_sums = np.asarray(aperture_photometry(variance, aperture, method="exact")["aperture_sum"], dtype=float)
    area = aperture.area_overlap(data) if hasattr(aperture, "area_overlap") else np.full(len(positions), aperture.area)
    area = np.asarray(area, dtype=float)
    flux = sums - sky_level*area
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_variance = variance_sums + (area*sky_standard_deviation)**2/np.where(number_of_sky_pixels > 0, number_of_sky_pixels, np.nan)
    return flux, np.sqrt(flux_variance), area


def quality_flags(image, positions, aperture_radius, outer_radius, peaks, neighbour_positions=None, neighbour_fluxes=None,
                  own_fluxes=None, crowding_radius=None, signal_to_noise=None):
    """Quality flags of each source (see FLAG_BITS)."""
    crowding_radius = outer_radius if crowding_radius is None else crowding_radius
    flags = np.zeros(len(positions), dtype=int)
    flags[np.asarray(peaks) > image.saturation_level] |= 1
    aperture = CircularAperture(positions, aperture_radius)
    mask_as_float = image.mask.astype(float)
    for index, aperture_mask in enumerate(aperture.to_mask(method="center")):
        values = aperture_mask.get_values(mask_as_float).astype(int)
        if np.any(values & (2 | 4)):
            flags[index] |= 2
        if np.any(values & 1):
            flags[index] |= 4
    number_of_rows, number_of_columns = image.data.shape
    x, y = positions[:, 0], positions[:, 1]
    flags[(x - outer_radius < 0) | (y - outer_radius < 0) | (x + outer_radius > number_of_columns - 1)
          | (y + outer_radius > number_of_rows - 1)] |= 8
    if neighbour_positions is not None and len(neighbour_positions) > 1:
        tree = cKDTree(neighbour_positions)
        for index, neighbours in enumerate(tree.query_ball_point(positions, crowding_radius)):
            for neighbour in neighbours:
                distance = np.hypot(*(neighbour_positions[neighbour] - positions[index]))
                if distance > 0.5 and neighbour_fluxes[neighbour] > 0.01*max(own_fluxes[index], 1e-30):
                    flags[index] |= 16
                    break
    if signal_to_noise is not None:
        flags[~(np.asarray(signal_to_noise) >= 5)] |= 32
    return flags


def background_noise_factor(image, radius, number_of_apertures=1000, source_mask=None, seed=0):
    """
    Ratio between the measured scatter of sums in empty-sky apertures and the scatter predicted by the VAR extension (>= 1).

    Two effects make it larger than 1: (1) the sub-pixel interpolation of cleaning.align() (or any resampling) correlates the noise of
    neighbouring pixels, so the per-pixel variance is right but the variance of a sum of pixels is underestimated; (2) in crowded fields,
    stars below the detection limit make the background fluctuate more than the photon noise (confusion noise), which is also a real error
    of aperture photometry. Only the faintest 30% of the background is used, to avoid gradients of diffuse light.
    Returns (factor, number of apertures used).
    """
    random_generator = np.random.default_rng(seed)
    number_of_rows, number_of_columns = image.data.shape
    margin = int(np.ceil(radius)) + 2
    if number_of_columns <= 2*margin or number_of_rows <= 2*margin:
        return 1.0, 0
    unusable = image.bad_pixels
    if source_mask is not None:
        unusable = unusable | source_mask
    positions = np.c_[random_generator.uniform(margin, number_of_columns - margin, 20*number_of_apertures),
                      random_generator.uniform(margin, number_of_rows - margin, 20*number_of_apertures)]
    aperture = CircularAperture(positions, radius)
    # "exact": an aperture that touches a bad pixel only partially is also rejected (otherwise a NaN pixel at its edge, e.g. in the
    # empty borders of aligned images, makes its sum NaN)
    unusable_fraction = np.asarray(aperture_photometry(unusable.astype(float), aperture, method="exact")["aperture_sum"], dtype=float)
    # Only in the faintest part of the background (the outskirts of a cluster or galaxy), where unresolved stars and gradients of diffuse
    # light do not add to the scatter: the factor should measure the pixel-to-pixel noise correlation only
    level = image.background[np.clip(positions[:, 1].astype(int), 0, number_of_rows - 1),
                             np.clip(positions[:, 0].astype(int), 0, number_of_columns - 1)]
    faint = level <= np.percentile(image.background, 30) + 0.5*float(np.median(image.background_rms))
    positions = positions[(unusable_fraction == 0) & faint][:number_of_apertures]
    if len(positions) < 50:
        return 1.0, len(positions)
    aperture = CircularAperture(positions, radius)
    residual = np.where(image.bad_pixels, 0.0, image.data - image.background)
    variance = np.where(image.bad_pixels | ~np.isfinite(image.variance), 0.0, np.clip(image.variance, 0, None))
    sums = np.asarray(aperture_photometry(residual, aperture, method="exact")["aperture_sum"], dtype=float)
    predicted = np.sqrt(np.asarray(aperture_photometry(variance, aperture, method="exact")["aperture_sum"], dtype=float))
    measured = 1.4826*np.median(np.abs(sums - np.median(sums)))
    return float(max(measured/np.median(predicted), 1.0)), int(len(positions))


def quick_aperture_fluxes(image, positions, radius, data=None):
    """Background-subtracted (background map) sums in circular apertures: a fast brightness estimate for crowding checks."""
    if len(positions) == 0:
        return np.zeros(0)
    if data is None:
        data = image.data - (image.background if image.background is not None else 0.0)
    data = np.where(np.isfinite(data), data, 0.0)
    table = aperture_photometry(data, CircularAperture(np.asarray(positions), radius), method="center")
    return np.clip(np.asarray(table["aperture_sum"], dtype=float), 0, None)


def exclusion_mask(shape, positions, radius):
    """Boolean image with discs of the given radius around the positions (used to keep stars out of the sky annuli)."""
    mask = np.zeros(shape, dtype=bool)
    if len(positions) == 0:
        return mask
    integer_radius = int(np.ceil(radius))
    y_grid, x_grid = np.mgrid[-integer_radius:integer_radius + 1, -integer_radius:integer_radius + 1]
    disc = (x_grid**2 + y_grid**2) <= radius**2
    number_of_rows, number_of_columns = shape
    for x, y in np.round(np.asarray(positions)).astype(int):
        x_start, x_end, y_start, y_end = x - integer_radius, x + integer_radius + 1, y - integer_radius, y + integer_radius + 1
        if x_end <= 0 or y_end <= 0 or x_start >= number_of_columns or y_start >= number_of_rows:
            continue
        clipped_x_start, clipped_y_start = max(x_start, 0), max(y_start, 0)
        clipped_x_end, clipped_y_end = min(x_end, number_of_columns), min(y_end, number_of_rows)
        mask[clipped_y_start:clipped_y_end, clipped_x_start:clipped_x_end] |= disc[clipped_y_start - y_start:clipped_y_end - y_start,
                                                                                  clipped_x_start - x_start:clipped_x_end - x_start]
    return mask


def isolation_mask(positions, own_fluxes, neighbour_positions, neighbour_fluxes, radius, maximum_ratio):
    """True for the sources with no other source closer than 'radius' brighter than maximum_ratio x their own flux."""
    isolated = np.ones(len(positions), dtype=bool)
    if neighbour_positions is None or len(neighbour_positions) < 2:
        return isolated
    tree = cKDTree(neighbour_positions)
    for index, neighbours in enumerate(tree.query_ball_point(positions, radius)):
        for neighbour in neighbours:
            if (np.hypot(*(neighbour_positions[neighbour] - positions[index])) > 0.5
                    and neighbour_fluxes[neighbour] > maximum_ratio*max(own_fluxes[index], 1e-30)):
                isolated[index] = False
                break
    return isolated


def robust_ratio_correction(ratio, small_flux, small_flux_error, large_flux_error, usable, maximum_number_of_stars=20):
    """
    Sigma-clipped median of 'ratio' (large over small flux) for the brightest usable stars, its uncertainty, the number of stars and the
    intrinsic star-to-star scatter (robust scatter minus the expected noise of the ratios, relative to the median).
    """
    used = np.flatnonzero(usable)
    used = used[np.argsort(-small_flux[used])][:maximum_number_of_stars]
    values = ratio[used]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, median, _ = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    # The flux in the small aperture is part of the flux in the large one: the noise of the ratio is ~ the noise of the area between them
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_noise = np.sqrt(np.clip(large_flux_error[used]**2 - small_flux_error[used]**2, 0, None))/small_flux[used]
    robust_scatter = 1.4826*np.median(np.abs(values - median))
    intrinsic_scatter = np.sqrt(max(robust_scatter**2 - np.nanmedian(ratio_noise)**2, 0.0))
    return float(median), float(1.2533*robust_scatter/np.sqrt(len(values))), int(len(values)), float(intrinsic_scatter/median)


def aperture_correction(image, positions, small_aperture_flux, small_aperture_error, flags, large_radius, sky_level,
                        sky_standard_deviation, number_of_sky_pixels, neighbour_positions=None, neighbour_fluxes=None, own_fluxes=None,
                        minimum_signal_to_noise=100.0, maximum_number_of_stars=20):
    """
    Aperture correction: sigma-clipped median ratio between the flux in a large aperture (large_radius, ~total) and in the photometry
    aperture, for the (up to maximum_number_of_stars) brightest unflagged and isolated stars. A star is isolated if no other detected
    source closer than large_radius + 2 FWHM is brighter than 0.1% of it (a neighbour with 1% of the flux inside the large aperture would
    already bias the correction by 1%). Faint undetected neighbours add a fixed amount of light to the large aperture, which biases the
    ratio of fainter stars upwards: only the brightest stars are used.
    Returns (correction, uncertainty of the median, number of stars, relative star-to-star scatter of the ratios).
    """
    fwhm = image.fwhm or 3.0
    own_fluxes = small_aperture_flux if own_fluxes is None else own_fluxes
    unflagged = (flags == 0)
    isolated = isolation_mask(positions, own_fluxes, neighbour_positions, neighbour_fluxes, large_radius + 2*fwhm, 1e-3)
    large_aperture_flux, large_aperture_error, _ = aperture_fluxes(image, positions, large_radius, sky_level, sky_standard_deviation,
                                                                   number_of_sky_pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal_to_noise = large_aperture_flux/large_aperture_error
        ratio = large_aperture_flux/small_aperture_flux
    plausible = np.isfinite(ratio) & (ratio > 0.8) & (ratio < 2.0)
    for selection in (unflagged & isolated & (signal_to_noise > minimum_signal_to_noise), unflagged & isolated & (signal_to_noise > 30),
                      unflagged & (signal_to_noise > minimum_signal_to_noise)):
        usable = selection & plausible
        if np.count_nonzero(usable) >= 3:
            break
    else:
        return 1.0, np.nan, 0, 0.0
    return robust_ratio_correction(ratio, small_aperture_flux, small_aperture_error, large_aperture_error, usable,
                                   maximum_number_of_stars)


# ======================================================================================================================================
# Matching
# ======================================================================================================================================

def estimate_translation(reference_positions, other_positions, maximum_offset=None, number_of_bright_sources=150, bin_size=1.0,
                         minimum_sources=3):
    """
    Translation (dx, dy) such that other_positions + (dx, dy) ~ reference_positions, from a vote on all pairwise differences of the
    brightest sources (both lists sorted by brightness), refined with the median of the matched pairs. Robust to unmatched sources.
    Returns dx, dy and the number of matched pairs.
    """
    reference = np.asarray(reference_positions, dtype=float)[:number_of_bright_sources]
    other = np.asarray(other_positions, dtype=float)[:number_of_bright_sources]
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    other = other[np.all(np.isfinite(other), axis=1)]
    if len(reference) < minimum_sources or len(other) < minimum_sources:
        return 0.0, 0.0, 0
    differences = (reference[:, None, :] - other[None, :, :]).reshape(-1, 2)
    # Without a limit, the offsets can be as large as the extent of the lists (capped to keep the histogram small)
    extent = np.ptp(np.r_[reference, other], axis=0).max() + bin_size
    limit = min(maximum_offset if maximum_offset is not None else extent, 4096)
    differences = differences[np.all(np.abs(differences) < limit, axis=1)]
    if len(differences) == 0:
        return 0.0, 0.0, 0
    bins = np.arange(-limit, limit + bin_size, bin_size)
    histogram, x_edges, y_edges = np.histogram2d(differences[:, 0], differences[:, 1], bins=[bins, bins])
    histogram = ndimage.uniform_filter(histogram, size=3)
    x_index, y_index = np.unravel_index(np.argmax(histogram), histogram.shape)
    dx, dy = 0.5*(x_edges[x_index] + x_edges[x_index + 1]), 0.5*(y_edges[y_index] + y_edges[y_index + 1])
    distance, index = cKDTree(reference).query(other + [dx, dy], distance_upper_bound=3*bin_size)
    matched = np.isfinite(distance)
    if np.count_nonzero(matched) < min(minimum_sources, 2):
        return 0.0, 0.0, 0                     # no two sources agree: the peak of the histogram is a chance coincidence
    dx, dy = np.median(reference[index[matched]] - other[matched], axis=0)
    return float(dx), float(dy), int(np.count_nonzero(matched))


def match_positions(reference_positions, other_positions, maximum_separation=2.0, offset=(0.0, 0.0)):
    """
    One-to-one nearest-neighbour match of other_positions + offset to reference_positions. Returns the indices in both lists and the
    separations.
    """
    tree = cKDTree(np.asarray(reference_positions))
    shifted = np.asarray(other_positions) + np.asarray(offset)
    distance, index = tree.query(np.nan_to_num(shifted, nan=-1e9), distance_upper_bound=maximum_separation)
    found = np.isfinite(distance)
    other_index, reference_index, separation = np.flatnonzero(found), index[found], distance[found]
    # One-to-one: keep the closest when several sources match the same reference
    order = np.argsort(separation)
    _, first_occurrence = np.unique(reference_index[order], return_index=True)
    keep = order[first_occurrence]
    return reference_index[keep], other_index[keep], separation[keep]


# ======================================================================================================================================
# Standard stars and calibration
# ======================================================================================================================================

STANDARD_STAR_CATALOGUE_DIRECTORY = Path(__file__).resolve().parent / "standard_star_catalogues"
STANDARD_STAR_CATALOGUE_ALIASES = {"landolt": ("landolt_2009", "landolt_2013"), "clem_landolt": ("clem_landolt_2013", "clem_landolt_2016")}


def available_standard_star_catalogues():
    """Names of the catalogues of standard stars distributed with easyspec (csv files in the standard_star_catalogues folder)."""
    return sorted(path.stem for path in STANDARD_STAR_CATALOGUE_DIRECTORY.glob("*.csv"))


def load_standard_star_catalogue(name):
    """
    A catalogue of standard stars distributed with easyspec, by name ("landolt_1992", "landolt_2009", "landolt_2013", "clem_landolt_2013",
    "clem_landolt_2016"). Aliases: "landolt" = Landolt (2009) + Landolt (2013); "clem_landolt" = the faint standards of Clem & Landolt
    (2013) + (2016). Any other catalogue can be given as a csv file. Returns a Table (columns name, field, ra, dec, magnitudes).
    """
    key = str(name).strip().lower()
    available = available_standard_star_catalogues()
    parts = [part for part in STANDARD_STAR_CATALOGUE_ALIASES.get(key, (key,)) if part in available]
    if not parts:
        hint = ("No catalogue of standard stars is installed: run easyspec/photometry/standard_star_catalogues/"
                "build_standard_star_catalogues.py once (it needs an internet connection)." if not available else
                f"Available catalogues: {', '.join(available)} (and the aliases {', '.join(STANDARD_STAR_CATALOGUE_ALIASES)}).")
        raise ValueError(f"Unknown catalogue of standard stars '{name}'. {hint}")
    tables = [Table.read(STANDARD_STAR_CATALOGUE_DIRECTORY / f"{part}.csv", format="ascii.csv") for part in parts]
    if len(tables) == 1:
        table = tables[0]
    else:
        from astropy.table import vstack
        for part, table in zip(parts, tables):
            table["catalogue"] = part
        table = vstack(tables, join_type="outer", metadata_conflicts="silent")
    table.meta["catalogue"] = key
    return table


def is_standard_star_catalogue_name(catalogue):
    """True if 'catalogue' is the name of a catalogue distributed with easyspec (and not a file)."""
    if not isinstance(catalogue, str) or Path(catalogue).suffix:
        return False
    key = catalogue.strip().lower()
    return key in STANDARD_STAR_CATALOGUE_ALIASES or key in available_standard_star_catalogues() or not Path(catalogue).exists()


def normalized_star_name(name):
    """
    Key to compare star names written in different ways: "SA 98-650", "SA98 650", "98 650" and "98-650" are the same star. Letters and numbers
    are kept in order (numbers without leading zeros); the prefix "SA" (Selected Area) is dropped.
    """
    tokens = re.findall(r"[A-Z]+|\d+", str(name).upper())
    tokens = [str(int(token)) if token.isdigit() else token for token in tokens]
    if tokens and tokens[0] == "SA":
        tokens = tokens[1:]
    return " ".join(tokens)


def read_catalogue(catalogue, require_coordinates=True):
    """
    Catalogue of standard (or reference) stars, as a Table with columns name, ra, dec (deg) and one column per band. Magnitudes can be given
    per band (e.g. B, V, R) and/or as a V magnitude with color indices (e.g. "B-V", "V-R", "R-I", "U-B"), as in Landolt's catalogues.
    'catalogue' is a csv file, a Table, a list of dicts, or the name of a catalogue distributed with easyspec (e.g. "landolt"; see
    available_standard_star_catalogues()). With require_coordinates=False, ra and dec are optional (lists of standards identified by hand).
    """
    if isinstance(catalogue, str) and is_standard_star_catalogue_name(catalogue):
        table = load_standard_star_catalogue(catalogue)
    elif isinstance(catalogue, (str, Path)):
        table = Table.read(catalogue, format="ascii.csv")
    elif isinstance(catalogue, (list, tuple)):
        table = Table(rows=list(catalogue)) if catalogue and isinstance(catalogue[0], dict) else Table(catalogue)
    else:
        table = Table(catalogue)
    new_names = {}
    for name in table.colnames:
        lower_case = name.lower()
        if lower_case in ("ra", "ra_deg", "raj2000", "_raj2000"):
            new_names[name] = "ra"
        elif lower_case in ("dec", "dec_deg", "dej2000", "_dej2000"):
            new_names[name] = "dec"
        elif lower_case in ("name", "id", "star"):
            new_names[name] = "name"
    for old_name, new_name in new_names.items():
        if old_name != new_name:
            table.rename_column(old_name, new_name)
    # Color indices: X-Y gives X from Y (or Y from X), repeated until nothing changes
    color_indices = [name for name in table.colnames if re.fullmatch(r"[A-Za-z]+-[A-Za-z]+", name)]
    changed = True
    while changed:
        changed = False
        for color_index in color_indices:
            first_band, second_band = color_index.split("-")
            if second_band in table.colnames and first_band not in table.colnames:
                table[first_band] = table[second_band] + table[color_index]
                changed = True
            elif first_band in table.colnames and second_band not in table.colnames:
                table[second_band] = table[first_band] - table[color_index]
                changed = True
    if require_coordinates and ("ra" not in table.colnames or "dec" not in table.colnames):
        raise ValueError("The catalogue needs the columns ra and dec (degrees).")
    if "name" not in table.colnames:
        table["name"] = [f"star_{index + 1}" for index in range(len(table))]
    table["name"] = [str(value) for value in table["name"]]
    return table


def fit_photometric_transformation(standard_magnitudes, instrumental_magnitudes, errors, airmasses, colors, fit_extinction=True,
                                   fit_color_term=True, fixed_extinction=None, fixed_color_term=None, clip_sigma=3.0, maximum_iterations=10):
    """
    Weighted least-squares fit of  M_standard - m_instrumental = ZP - k X - c color  with iterative sigma clipping. Coefficients that are
    not fitted are fixed to fixed_extinction and fixed_color_term. The covariance is scaled by the reduced chi2 when it is larger than 1
    (the scatter of standard-star photometry is usually larger than the formal errors). Returns a dict.
    """
    difference = np.asarray(standard_magnitudes) - np.asarray(instrumental_magnitudes)
    airmasses, colors = np.asarray(airmasses, dtype=float), np.asarray(colors, dtype=float)
    sigma = np.clip(np.asarray(errors, dtype=float), 0.005, None)    # a floor: no star is better than 5 mmag
    fitted_quantity = difference.copy()
    if not fit_extinction:
        fitted_quantity = fitted_quantity + (fixed_extinction or 0.0)*airmasses
    if not fit_color_term:
        fitted_quantity = fitted_quantity + (fixed_color_term or 0.0)*colors
    columns, parameter_names = [np.ones_like(airmasses)], ["ZP"]
    if fit_extinction:
        columns.append(-airmasses)
        parameter_names.append("k")
    if fit_color_term:
        columns.append(-colors)
        parameter_names.append("c")
    design_matrix = np.vstack(columns).T
    number_of_parameters = design_matrix.shape[1]
    used = np.isfinite(fitted_quantity) & np.isfinite(sigma)
    if np.count_nonzero(used) < number_of_parameters:
        raise ValueError(f"Not enough standard-star measurements for the fit: {np.count_nonzero(used)} for {number_of_parameters} free "
                         f"parameters ({', '.join(parameter_names)}). Fix the color term (fit_color_term=False) and/or the extinction "
                         "(fit_extinction=False), or observe more standards.")
    if np.count_nonzero(used) == number_of_parameters:
        warnings.warn(f"Only {number_of_parameters} standard-star measurement(s) for {number_of_parameters} free parameter(s): the "
                      "solution is exact (no outlier can be detected) and its errors come only from the photometric errors of these "
                      "measurements, not from the scatter of the standards.")
    for _ in range(maximum_iterations):
        weights = 1/sigma[used]
        coefficients, *_ = np.linalg.lstsq(design_matrix[used]*weights[:, None], fitted_quantity[used]*weights, rcond=None)
        residuals = fitted_quantity - design_matrix @ coefficients
        if np.count_nonzero(used) <= number_of_parameters + 2:
            break                                         # too few measurements to recognize outliers
        robust_scatter = 1.4826*np.median(np.abs(residuals[used] - np.median(residuals[used])))
        new_used = used & (np.abs(residuals) < clip_sigma*np.sqrt(robust_scatter**2 + sigma**2))
        if np.array_equal(new_used, used) or np.count_nonzero(new_used) < number_of_parameters:
            break
        used = new_used
    weights = 1/sigma[used]
    degrees_of_freedom = np.count_nonzero(used) - number_of_parameters
    reduced_chi2 = float(np.sum((residuals[used]*weights)**2)/degrees_of_freedom) if degrees_of_freedom > 0 else np.nan
    weighted_design = design_matrix[used]*weights[:, None]
    covariance = np.linalg.inv(weighted_design.T @ weighted_design)*(max(reduced_chi2, 1.0) if degrees_of_freedom > 0 else 1.0)
    values = {"ZP": np.nan, "k": fixed_extinction if not fit_extinction else np.nan, "c": fixed_color_term if not fit_color_term else np.nan}
    uncertainties = {"ZP": np.nan, "k": 0.0 if not fit_extinction else np.nan, "c": 0.0 if not fit_color_term else np.nan}
    for name, value, uncertainty in zip(parameter_names, coefficients, np.sqrt(np.diag(covariance))):
        values[name], uncertainties[name] = float(value), float(uncertainty)
    full_covariance = np.zeros((3, 3))
    position_of = {"ZP": 0, "k": 1, "c": 2}
    for row, name_row in enumerate(parameter_names):
        for column, name_column in enumerate(parameter_names):
            full_covariance[position_of[name_row], position_of[name_column]] = covariance[row, column]
    return {"coefficients": values, "errors": uncertainties, "covariance": full_covariance, "rms": float(np.std(residuals[used])),
            "reduced_chi2": reduced_chi2, "number_used": int(np.count_nonzero(used)), "number_total": int(len(difference)), "used": used,
            "residuals": residuals}


def solve_two_band_magnitudes(instrumental_1, instrumental_2, airmass_1, airmass_2, solution_1, solution_2, error_1=None, error_2=None):
    """
    Standard magnitudes M1, M2 of the bands 1 and 2 (color index M1 - M2) from instrumental magnitudes m1, m2 at airmasses X1, X2:
        M1 = m1 + ZP1 - k1 X1 - c1 (M1 - M2),   M2 = m2 + ZP2 - k2 X2 - c2 (M1 - M2).
    The color is solved exactly: (M1 - M2)(1 + c1 - c2) = (m1 - m2) + (ZP1 - ZP2) - k1 X1 + k2 X2.
    Returns M1, M2, color, their statistical errors (from error_1, error_2) and their calibration errors (from the coefficient covariances).
    """
    coefficients_1, coefficients_2 = solution_1["coefficients"], solution_2["coefficients"]
    denominator = 1 + coefficients_1["c"] - coefficients_2["c"]
    color = ((instrumental_1 - instrumental_2) + (coefficients_1["ZP"] - coefficients_2["ZP"]) - coefficients_1["k"]*airmass_1
             + coefficients_2["k"]*airmass_2)/denominator
    magnitude_1 = instrumental_1 + coefficients_1["ZP"] - coefficients_1["k"]*airmass_1 - coefficients_1["c"]*color
    magnitude_2 = instrumental_2 + coefficients_2["ZP"] - coefficients_2["k"]*airmass_2 - coefficients_2["c"]*color
    # Statistical errors: derivatives with respect to m1 and m2
    error_1 = np.zeros_like(instrumental_1) if error_1 is None else error_1
    error_2 = np.zeros_like(instrumental_2) if error_2 is None else error_2
    derivative_1_by_1, derivative_1_by_2 = 1 - coefficients_1["c"]/denominator, coefficients_1["c"]/denominator
    derivative_2_by_1, derivative_2_by_2 = -coefficients_2["c"]/denominator, 1 + coefficients_2["c"]/denominator
    magnitude_1_error = np.hypot(derivative_1_by_1*error_1, derivative_1_by_2*error_2)
    magnitude_2_error = np.hypot(derivative_2_by_1*error_1, derivative_2_by_2*error_2)
    color_error = np.hypot(error_1, error_2)/abs(denominator)
    # Calibration errors (Monte Carlo over the coefficients, which are correlated within each band)
    random_generator = np.random.default_rng(1)
    draws_1 = random_generator.multivariate_normal([coefficients_1["ZP"], coefficients_1["k"], coefficients_1["c"]],
                                                   solution_1["covariance"], size=400)
    draws_2 = random_generator.multivariate_normal([coefficients_2["ZP"], coefficients_2["k"], coefficients_2["c"]],
                                                   solution_2["covariance"], size=400)
    typical_1, typical_2 = np.nanmedian(instrumental_1), np.nanmedian(instrumental_2)
    typical_airmass_1, typical_airmass_2 = np.nanmedian(airmass_1), np.nanmedian(airmass_2)
    samples = []
    for (zero_point_1, extinction_1, color_term_1), (zero_point_2, extinction_2, color_term_2) in zip(draws_1, draws_2):
        sample_denominator = 1 + color_term_1 - color_term_2
        sample_color = ((typical_1 - typical_2) + (zero_point_1 - zero_point_2) - extinction_1*typical_airmass_1
                        + extinction_2*typical_airmass_2)/sample_denominator
        samples.append((typical_1 + zero_point_1 - extinction_1*typical_airmass_1 - color_term_1*sample_color,
                        typical_2 + zero_point_2 - extinction_2*typical_airmass_2 - color_term_2*sample_color, sample_color))
    calibration_errors = np.std(np.array(samples), axis=0)
    return magnitude_1, magnitude_2, color, magnitude_1_error, magnitude_2_error, color_error, calibration_errors
