"""
easyspec color_composite module.

Helpers of cleaning.generate_colored_image(): reading the band images, removing the stars with PSF photometry, normalizing and
stretching the bands, mixing them with arbitrary colors, and saving the result (png or fits). Users normally do not need to call these
functions directly.
"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy import ndimage
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from .. import data_io


VALID_COLOR_MODES = ("channels", "preserve")
VALID_BAND_SCALINGS = ("separate", "exposure")


VALID_STRETCHES = ("asinh", "log", "linear")


def read_band_images(images):
    """
    Band images from file paths (reduced frames saved by cleaning.save_fits_files() or any fits image) or 2D arrays.
    Returns names, data arrays, variance arrays (None if unknown), masks (None if unknown) and headers (None for arrays).
    """
    names, data, variances, masks, headers = [], [], [], [], []
    for index, image in enumerate(images):
        if isinstance(image, (str, Path)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame, variance, mask, header = data_io.read_reduced_frame(str(image))
            names.append(str(Path(image).resolve()))
            data.append(np.asarray(frame, dtype=float))
            variances.append(None if variance is None else np.asarray(variance, dtype=float))
            masks.append(mask)
            headers.append(header)
        else:
            array = np.asarray(image, dtype=float)
            if array.ndim != 2:
                raise ValueError(f"Image {index} is not a 2D array (shape {array.shape}).")
            names.append(f"image_{index}")
            data.append(array)
            variances.append(None)
            masks.append(None)
            headers.append(None)
    return names, data, variances, masks, headers


def fit_stars_with_psf(name, data, variance, mask, header, saturation_level=None, fwhm=None, verbose=True):
    """
    PSF fit of the stars of one band, by easyspec.photometry.psf_photometry() (empirical PSF). Nothing is subtracted here: the stars to
    remove are chosen later (those found in all bands, see match_stars_in_all_bands()) and subtracted by subtract_stars().
    Returns a dictionary with the table of stars ("table": x, y in 0-based pixels, flux...), the PSF model ("psf_model",
    "psf_stamp_size"), the FWHM in pixels ("fwhm"), the saturation level used ("saturation_level") and the bad pixels ("bad_pixels").
    """
    from ..photometry.photometry import photometry
    from ..photometry import aux_photometry
    if variance is None:
        variance = np.clip(data, 0, None) + 1.0
    if mask is None:
        mask = np.zeros(data.shape, dtype=np.uint8)
    image = aux_photometry.Image(name=Path(name).name, data=data.copy(), variance=variance, mask=mask,
                                 header=header if header is not None else fits.Header(),
                                 saturation_level=saturation_level if saturation_level is not None else np.inf,
                                 exposure_time=float(header.get("EXPTIME", 1.0)) if header is not None else 1.0)
    if saturation_level is None and header is not None and "SATURATE" in header:
        image.saturation_level = 0.85*float(header["SATURATE"])*float(header.get("GAIN", 1.0))
    phot = photometry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = phot.psf_photometry([image], fwhm=fwhm if fwhm is not None else "auto", plot=False, verbose=verbose)
    return {"table": tables[0], "psf_model": image.extra["psf_model"], "psf_stamp_size": image.extra["psf_stamp_size"],
            "fwhm": image.fwhm or 3.0, "saturation_level": image.saturation_level, "bad_pixels": image.bad_pixels,
            "variance": variance}


def match_stars_in_all_bands(positions, radius):
    """
    Stars found in all bands. positions: one (N, 2) array of (x, y) per band, all in the pixels of the reference image. A star of the
    first band is kept if every other band has a star within 'radius' pixels that is, in turn, closest to it (mutual nearest
    neighbours). Returns, for each band, the indices (in that band's list) of the common stars, in the same order in all bands.
    """
    from scipy.spatial import cKDTree
    positions = [np.where(np.isfinite(np.asarray(xy, dtype=float)), np.asarray(xy, dtype=float), 1e12).reshape(-1, 2)
                 for xy in positions]
    reference = positions[0]
    empty = [np.array([], dtype=int) for _ in positions]
    if any(len(xy) == 0 for xy in positions):
        return empty
    reference_tree = cKDTree(reference)
    keep = np.ones(len(reference), dtype=bool)
    matches = [np.arange(len(reference))]
    for other in positions[1:]:
        distance, index = cKDTree(other).query(reference, distance_upper_bound=radius)
        found = np.isfinite(distance)
        index = np.where(found, index, 0)
        _, back = reference_tree.query(other[index])                    # the nearest reference star of that star
        found &= back == np.arange(len(reference))
        keep &= found
        matches.append(index)
    return [match[keep] for match in matches]


def star_model_from_table(table, psf_model, stamp_size, shape):
    """
    Image of the PSF models of the stars of a psf_photometry() table (columns x, y and flux, in 0-based pixels).
    """
    model = np.zeros(shape, dtype=float)
    half = int(stamp_size)//2
    for x, y, flux in zip(np.asarray(table["x"], dtype=float), np.asarray(table["y"], dtype=float), np.asarray(table["flux"], dtype=float)):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(flux)):
            continue
        column, row = int(round(x)), int(round(y))
        x1, x2 = max(column - half, 0), min(column + half + 1, shape[1])
        y1, y2 = max(row - half, 0), min(row + half + 1, shape[0])
        if x1 >= x2 or y1 >= y2:
            continue
        grid_y, grid_x = np.mgrid[y1:y2, x1:x2]
        model[y1:y2, x1:x2] += psf_model.evaluate(grid_x, grid_y, flux, x, y)
    return model


def subtract_stars(data, fit, selected):
    """
    Image without the selected stars (indices in fit["table"], from match_stars_in_all_bands()) and the pixels where the subtraction
    cannot be trusted: saturated stars (whose cores are clipped), enlarged by 3 FWHM, and the cores of bright stars where the residual is
    much larger than the noise (including a 5% uncertainty of the PSF model), enlarged by 1 FWHM. They are filled later by
    fill_unreliable_pixels(), with the same mask in all bands. Returns (residual image, unreliable pixels).
    """
    variance = fit["variance"]
    model = star_model_from_table(fit["table"][np.asarray(selected, dtype=int)], fit["psf_model"], fit["psf_stamp_size"], data.shape)
    residual = data - model
    fwhm_pixels = fit["fwhm"]
    noise = np.sqrt(np.clip(variance, 0, None) + (0.05*model)**2)
    # only the saturated or bad pixels under the removed stars (a star that is not removed is left as it is)
    saturated = ((data >= fit["saturation_level"]) | fit["bad_pixels"]) & (model > 0)
    bright_cores = (model > 5*np.sqrt(np.clip(variance, 1e-12, None))) & (np.abs(residual) > 3*noise)
    unreliable = (ndimage.binary_dilation(saturated, iterations=max(int(round(3*fwhm_pixels)), 1))
                  | ndimage.binary_dilation(bright_cores, iterations=max(int(round(fwhm_pixels)), 1)))
    return residual, unreliable


def fill_unreliable_pixels(bands, unreliable, smoothing_sigma, seed=0):
    """
    Replace the unreliable pixels (the same in all bands, so that the filled patches have consistent colors) by the surrounding image
    (normalized Gaussian interpolation) plus noise with the local scatter of the image.
    """
    random = np.random.default_rng(seed)
    filled_bands = []
    for band in bands:
        good = (~unreliable & np.isfinite(band)).astype(float)
        values = np.where(good > 0, band, 0.0)
        weight = np.maximum(ndimage.gaussian_filter(good, smoothing_sigma), 1e-6)
        smooth = ndimage.gaussian_filter(values, smoothing_sigma)/weight
        local_scatter = np.sqrt(np.clip(ndimage.gaussian_filter(np.where(good > 0, (band - smooth)**2, 0.0), smoothing_sigma)/weight, 0, None))
        filled = band.copy()
        replace = unreliable & np.isfinite(band)
        filled[replace] = smooth[replace] + random.normal(0.0, 1.0, np.count_nonzero(replace))*local_scatter[replace]
        filled_bands.append(filled)
    return filled_bands


def estimate_saturation_level(image, header=None):
    """
    Saturation level of an image, in its own units. From the SATURATE keyword (times GAIN if the image was converted to electrons) when
    present; otherwise from the data: saturated pixels pile up at the same value (the top of the analog-to-digital converter, slightly
    spread by the flat field), which shows as a peak at the bright end of the histogram. Returns None if no such pile is found.
    """
    if header is not None and "SATURATE" in header:
        level = float(header["SATURATE"])
        if str(header.get("BUNIT", "")).strip().lower().startswith("electron") or header.get("GAINCORR", False):
            level *= float(header.get("GAIN", 1.0))
        return 0.95*level
    values = image[np.isfinite(image)]
    values = values[values > 0]
    if values.size < 100:
        return None
    top = values.max()
    bright = values[values >= max(np.percentile(values, 90), top/50)]
    edges = np.geomspace(bright.min(), top*1.0001, int(np.log(top*1.0001/bright.min())/np.log(1.01)) + 2)    # 1% wide bins
    counts, edges = np.histogram(bright, bins=edges)
    # a pile: many pixels in one bin, many more than in the bins just below it (the wings of the stars decrease smoothly)
    best, best_count = None, 0
    for index in range(len(counts)):
        below = counts[max(index - 6, 0):max(index - 2, 0)]
        reference = np.median(below) if below.size else 0
        if counts[index] >= 20 and counts[index] >= 3*max(reference, 1) and counts[index] > best_count:
            best, best_count = index, counts[index]
    if best is None:
        return None
    return 0.95*edges[best]


def remove_saturation_trails(image, saturation_level, maximum_extension=600, seed=0):
    """
    Cosmetic removal of the bleeding trails of saturated stars (the charge that overflows along the columns, or rows, of the CCD), for
    pretty pictures only: the photometry of these pixels is meaningless anyway. For each saturated region, the round core of the star
    (as wide as the widest part of the region) is kept; the trail on both sides of the core (as narrow as the saturated trail, extended
    while it differs from the pixels beside it) is replaced, line by line, by an interpolation between the pixels on both sides of the
    trail, plus noise like that of the surroundings. Stars crossed by the trail are kept as far as their sides allow.
    Returns the cleaned image and the mask of the replaced pixels.
    """
    random = np.random.default_rng(seed)
    cleaned = image.copy()
    finite = np.isfinite(image)
    _, background, background_rms = sigma_clipped_stats(image[finite], sigma=3.0, maxiters=5)
    saturated = finite & (image >= saturation_level)
    # pixels next to saturated ones with almost no signal (the readout of saturated columns undershoots): replaced by the median of
    # their neighbours
    near = ndimage.binary_dilation(saturated, iterations=3) & ~saturated
    dead = near & ~(image > 0.05*saturation_level)
    if np.any(dead):
        with_gaps = np.where(dead | ~finite, np.nan, image)
        for row, column in zip(*np.nonzero(dead)):
            window = with_gaps[max(row - 2, 0):row + 3, max(column - 2, 0):column + 3]
            if np.any(np.isfinite(window)):
                cleaned[row, column] = np.nanmedian(window)
        image = cleaned.copy()
    labels, _ = ndimage.label(ndimage.binary_dilation(saturated, iterations=1))
    trails = dead.copy()
    for label_index, region in enumerate(ndimage.find_objects(labels), start=1):
        rows, columns = np.nonzero(labels[region] == label_index)
        rows, columns = rows + region[0].start, columns + region[1].start
        vertical = np.ptp(rows) >= np.ptp(columns)
        along, across = (rows, columns) if vertical else (columns, rows)
        # width of the region at each position along the trail: the core of the star is its widest part
        positions = np.arange(along.min(), along.max() + 1)
        widths = np.array([np.ptp(across[along == position]) + 1 if np.any(along == position) else 0 for position in positions])
        core_index = int(np.argmax(widths))
        core_along = positions[core_index]
        core_across = 0.5*(across[along == core_along].min() + across[along == core_along].max())
        core_radius = 0.6*widths[core_index] + 1
        # dead or wrapped pixels inside the core of the star: set to the saturation level
        box = (slice(max(int(rows.min()) - 3, 0), min(int(rows.max()) + 4, image.shape[0])),
               slice(max(int(columns.min()) - 3, 0), min(int(columns.max()) + 4, image.shape[1])))
        box_y, box_x = np.mgrid[box]
        box_along, box_across = (box_y, box_x) if vertical else (box_x, box_y)
        in_core = (box_along - core_along)**2 + (box_across - core_across)**2 <= (core_radius + 3)**2
        cleaned[box][in_core & ~(image[box] > background - 5*background_rms)] = saturation_level
        outside_core = np.abs(along - core_along) > core_radius
        if np.count_nonzero(outside_core) < 3:
            continue                                                        # no trail: a round saturated core only
        # the trail: as wide as the saturated pixels outside the core, plus 2 pixels on each side
        low = int(np.floor(np.percentile(across[outside_core], 2))) - 2
        high = int(np.ceil(np.percentile(across[outside_core], 98))) + 3
        length = image.shape[0] if vertical else image.shape[1]
        width = image.shape[1] if vertical else image.shape[0]
        low, high = max(low, 0), min(high, width)
        if high - low < 1:
            continue

        def line_at(position):
            return image[position] if vertical else image[:, position]

        def sides_of(line):
            values = np.r_[line[max(low - 6, 0):low], line[high:high + 6]]
            return values[np.isfinite(values)]

        # extend the trail beyond the saturated pixels, while it differs from the pixels beside it (bright charge or dead pixels)
        start, stop = int(along.min()), int(along.max())
        for direction in (-1, 1):
            position, steps = (start if direction < 0 else stop) + direction, 0
            while 0 <= position < length and steps < maximum_extension:
                line = line_at(position)
                sides = sides_of(line)
                if sides.size == 0 or np.nanmax(np.abs(line[low:high] - np.median(sides))) <= 5*background_rms:
                    break
                position, steps = position + direction, steps + 1
            if direction < 0:
                start = max(position, 0)
            else:
                stop = min(position, length - 1)
        # replace the trail line by line (the lines inside the core of the star are kept)
        work = cleaned if vertical else cleaned.T
        trail = trails if vertical else trails.T
        for position in range(start, stop + 1):
            line_across = np.arange(low, high)
            keep = (position - core_along)**2 + (line_across - core_across)**2 <= core_radius**2
            replace = line_across[~keep]
            if replace.size == 0:
                continue
            sides = (np.median(work[position, max(low - 4, 0):low]) if low > 0 else np.nan,
                     np.median(work[position, high:high + 4]) if high < width else np.nan)
            left, right = [value if np.isfinite(value) else np.nan for value in sides]
            if not np.isfinite(left) and not np.isfinite(right):
                continue
            left = right if not np.isfinite(left) else left
            right = left if not np.isfinite(right) else right
            values = np.interp(replace, [low - 2.5, high + 1.5], [left, right])
            noise = background_rms*np.sqrt(1 + np.clip(values - background, 0, None)/max(abs(background), background_rms, 1e-12))
            work[position, replace] = values + random.normal(0.0, 1.0, replace.size)*noise
            trail[position, replace] = True
    return cleaned, trails


def per_band(value, number_of_bands, name):
    """A scalar is used for all bands; a list must have one value per band."""
    if np.ndim(value) == 0:
        return [value]*number_of_bands
    if len(value) != number_of_bands:
        raise ValueError(f"Give one value of {name} per image ({number_of_bands}), or a single value for all of them.")
    return list(value)


def normalize_band(image, valid, white_percentile, black_level=0.0):
    """
    Background-subtracted band divided by its white point (percentile of the valid pixels), so that all bands have similar levels.
    black_level is in units of the background noise: positive values hide the noise of the sky (darker background), negative values
    lift the sky (greyer background).
    """
    _, background, background_rms = sigma_clipped_stats(image[valid], sigma=3.0, maxiters=5)
    black = background + black_level*background_rms
    white = np.nanpercentile(image[valid] - black, white_percentile)
    if not np.isfinite(white) or white <= 0:
        white = np.nanmax(image[valid] - black)
    return (image - black)/max(white, 1e-12)


def normalize_bands_by_flux(bands, valid, exposure_times, zero_points, white_percentile, black_levels):
    """
    Bands on a common flux scale: each band is background-subtracted and divided by its exposure time (count rate), and, if zero points
    are given (m = zero_point - 2.5 log10(count rate)), converted to flux relative to Vega (an A0 star has the same flux in all bands and
    looks white). All bands share the same white point (a percentile of the brightest band at each pixel), so the colors show the real
    flux ratios between the bands.
    """
    rates, blacks = [], []
    for index, band in enumerate(bands):
        _, background, background_rms = sigma_clipped_stats(band[valid], sigma=3.0, maxiters=5)
        factor = 1.0/exposure_times[index]
        if zero_points is not None:
            factor *= 10**(-0.4*zero_points[index])
        rates.append((band - background)*factor)
        blacks.append(black_levels[index]*background_rms*factor)
    brightest = np.max([rate[valid] for rate in rates], axis=0)
    white = np.nanpercentile(brightest, white_percentile)
    if not np.isfinite(white) or white <= 0:
        white = np.nanmax(brightest)
    return [(rate - black)/max(white, 1e-12) for rate, black in zip(rates, blacks)]


def smooth_band(image, smoothing_sigma):
    """Gaussian smoothing (sigma in pixels) that ignores NaN pixels: reduces the noise of faint diffuse light."""
    if not smoothing_sigma:
        return image
    good = np.isfinite(image)
    weight = np.maximum(ndimage.gaussian_filter(good.astype(float), smoothing_sigma), 1e-6)
    smooth = ndimage.gaussian_filter(np.where(good, image, 0.0), smoothing_sigma)/weight
    return np.where(good, smooth, np.nan)


def mix_colors(normalized_bands, colors):
    """
    Linear RGB image: sum of the normalized bands, each multiplied by its color (r, g, b), with every channel divided by the sum of
    the colors in that channel (so that e.g. three bands in blue, green and red, or any set of colors adding up to white, give
    neutral grey for a source with the same normalized level in all bands).
    """
    rgb_colors = np.array([to_rgb(color) for color in colors])
    shape = normalized_bands[0].shape
    rgb = np.zeros(shape + (3,))
    for band, color in zip(normalized_bands, rgb_colors):
        rgb += band[..., None]*color[None, None, :]
    channel_weight = rgb_colors.sum(axis=0)
    channel_weight[channel_weight == 0] = 1.0
    return rgb/channel_weight[None, None, :]


def stretch_rgb(rgb, stretch="log", strength=None, color_mode="channels"):
    """
    Stretch of a linear RGB image (1 = white point).
    color_mode = "channels": each channel is stretched separately, as in DS9 (rgb frames) and most image viewers; the cores of bright
    stars turn white and the faint parts keep their colors.
    color_mode = "preserve": the intensity I = mean(R, G, B) is stretched and the three channels are multiplied by the same factor f(I)/I
    (Lupton et al. 2004): the color of every pixel is kept, even in the brightest stars (pixels where a channel exceeds 1 are scaled down
    as a whole).
    strength: a for the log stretch log10(1 + a x)/log10(1 + a) (default 1000, as in DS9) and for the asinh stretch
    asinh(a x)/asinh(a) (default 100).
    """
    if stretch not in VALID_STRETCHES:
        raise ValueError(f"Invalid stretch '{stretch}'. Options are: " + ", ".join(f'"{name}"' for name in VALID_STRETCHES) + ".")
    if color_mode not in ("channels", "preserve"):
        raise ValueError("color_mode must be 'channels' or 'preserve'.")
    if strength is None:
        strength = {"log": 1000.0, "asinh": 100.0, "linear": 1.0}[stretch]

    def function(values):
        values = np.clip(values, 0, None)
        if stretch == "asinh":
            return np.arcsinh(strength*values)/np.arcsinh(strength)
        if stretch == "log":
            return np.log10(1 + strength*values)/np.log10(1 + strength)
        return values

    rgb = np.where(np.isfinite(rgb), rgb, 0.0)
    if color_mode == "channels":
        return np.clip(function(np.clip(rgb, 0, 1)), 0, 1)
    intensity = rgb.mean(axis=2)
    stretched = function(intensity)
    factor = np.where(intensity > 0, stretched/np.where(intensity > 0, intensity, 1.0), 0.0)
    result = np.clip(rgb*factor[..., None], 0, None)
    peak = result.max(axis=2)
    result = np.where(peak[..., None] > 1, result/np.maximum(peak, 1e-12)[..., None], result)
    return np.clip(result, 0, 1)


def adjust_rgb(rgb, gamma=1.0, color_saturation=1.0, brightness=1.0, contrast=1.0):
    """
    Final cosmetic adjustments of a stretched RGB image (values 0-1): color saturation (0 = grey scale, 1 = unchanged, > 1 = more
    vivid), gamma (< 1 brightens the faint parts, > 1 darkens them), contrast (around the mid grey 0.5) and brightness (multiplicative).
    """
    grey = rgb.mean(axis=2, keepdims=True)
    rgb = grey + color_saturation*(rgb - grey)
    rgb = np.clip(rgb, 0, 1)**gamma
    rgb = (rgb - 0.5)*contrast + 0.5
    return np.clip(rgb*brightness, 0, 1)


def show(rgb, title, crop_box=None, figure_size=None, show_axes=True, background_color=None):
    """Plot of an RGB image (origin at the bottom left), optionally with the crop box."""
    from matplotlib.patches import Rectangle
    if figure_size is None:
        figure_size = (10, 10*rgb.shape[0]/rgb.shape[1] + 0.6)
    figure, axis = plt.subplots(figsize=figure_size)
    if background_color is not None:
        figure.patch.set_facecolor(background_color)
    axis.imshow(rgb, origin="lower", interpolation="nearest")
    if crop_box is not None:
        x1, x2, y1, y2 = crop_box
        axis.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, edgecolor="white", linestyle="--", linewidth=1.5,
                                 label="final crop"))
        axis.legend(loc="upper right", fontsize=9, framealpha=0.6)
    if title:
        axis.set_title(title, fontsize=12)
    if show_axes:
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("y (pixels)")
    else:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def save(rgb, output_file, file_format=None, header=None, crop_box=None, colors=None, names=None):
    """Save the RGB image as png (8 bits per channel) or fits (cube of 3 planes R, G, B, with the WCS of the reference image)."""
    output_file = Path(output_file)
    file_format = (file_format or output_file.suffix.lstrip(".") or "png").lower()
    if file_format in ("fit", "fts"):
        file_format = "fits"
    if file_format not in ("png", "fits"):
        raise ValueError("The colored image can be saved as 'png' or 'fits'.")
    if output_file.suffix.lower().lstrip(".") != file_format and not (file_format == "fits" and output_file.suffix.lower() in (".fit", ".fts")):
        output_file = output_file.with_suffix("." + file_format)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "png":
        plt.imsave(output_file, np.flipud(rgb))                       # images are displayed with the origin at the bottom
    else:
        cube = np.moveaxis(rgb, 2, 0).astype(np.float32)               # (3, rows, columns): planes R, G, B
        new_header = fits.Header()
        if header is not None:
            try:
                from astropy.wcs import WCS
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wcs = WCS(header).celestial
                if wcs.has_celestial:
                    new_header.update(wcs.to_header())
                    if crop_box is not None:
                        new_header["CRPIX1"] -= crop_box[0]
                        new_header["CRPIX2"] -= crop_box[2]
            except Exception:
                pass
            for keyword in ("OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS"):
                if keyword in header:
                    new_header[keyword] = header[keyword]
        new_header["CTYPE3"] = ("RGB", "planes: red, green, blue")
        for index, (name, color) in enumerate(zip(names or [], colors or []), start=1):
            new_header[f"BAND{index}"] = (Path(name).name[:60], "input image")
            new_header[f"COLOR{index}"] = (str(color)[:60], "color of this image in the composite")
        new_header["COMMENT"] = "Color composite made with easyspec cleaning.generate_colored_image()"
        fits.PrimaryHDU(cube, header=new_header).writeto(output_file, overwrite=True)
    return output_file
