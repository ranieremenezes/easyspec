"""
easyspec.photometry: photometry of images reduced with easyspec.cleaning.

The reduction (bias, flat field, gain, cosmic rays, variance, alignment and combination) is done with easyspec.cleaning; this module starts
from the reduced images (in electrons, with the VAR and MASK extensions written by cleaning.save_fits_files() or cleaning.master()).

It offers:
    - aperture photometry (local sky, aperture correction, quality flags);
    - PSF ("profile") photometry with an empirical PSF and iterative detection of the stars hidden in crowded fields;
    - calibration with standard stars (zero point, extinction and color terms) and color-magnitude diagrams;
    - surface photometry of extended objects (surface-brightness profiles, isophotes, King and Sersic fits);
    - differential photometry of time series and the fit of exoplanet transits.

Typical workflow (calibrated aperture photometry of a cluster):

    from easyspec.photometry import photometry
    phot = photometry()
    standards = phot.load_images(["reduced/standards_B", "reduced/standards_V"])
    standard_tables = phot.aperture_photometry(standards)
    solution = phot.fit_standard_star_solution(standard_tables, catalogue="landolt", color_index=("B", "V"))   # images with WCS
    # without WCS, give the standards by hand: fit_standard_star_solution(standard_tables, standard_stars=..., ...)
    cluster = phot.load_images(["master_target_open_cluster_B.fits", "master_target_open_cluster_V.fits"])
    catalogue = phot.calibrate_magnitudes(phot.aperture_photometry(cluster), solution)
    phot.plot_color_magnitude_diagram(catalogue)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack

from . import aux_differential_photometry as aux_differential
from . import aux_photometry as aux
from . import aux_psf_photometry as aux_psf
from . import aux_surface_photometry as aux_surface
from . import aux_transit_model as aux_transit
from .aux_photometry import FLAG_BITS, MAGNITUDE_ERROR_FACTOR


def _image_normalization(data, image_scale="log", percentile_interval=99.5, log_strength=1000):
    """
    Display normalization of an image: "log" (default; shows faint stars and bright cores at the same time) or "linear", between the
    percentiles that keep percentile_interval % of the pixels.
    """
    from astropy.visualization import ImageNormalize, LinearStretch, LogStretch, PercentileInterval
    if image_scale not in ("log", "linear"):
        raise ValueError("image_scale must be 'log' or 'linear'.")
    finite = np.asarray(data)[np.isfinite(data)]
    stretch = LogStretch(a=log_strength) if image_scale == "log" else LinearStretch()
    return ImageNormalize(finite, interval=PercentileInterval(percentile_interval), stretch=stretch)


class photometry:

    """
    Photometry of images reduced with easyspec.cleaning: star detection, aperture and PSF photometry, calibration with standard stars,
    color-magnitude diagrams, surface photometry of extended objects, and differential photometry of time series (exoplanet transits).
    """

    FLAG_BITS = FLAG_BITS

    def __init__(self):
        self.images = []
        self.standard_star_solution = None

    # ==================================================================================================================================
    # Input
    # ==================================================================================================================================

    def load_images(self, source, band_header_entry="FILTER", exposure_header_entry="EXPTIME", airmass_header_entry="AIRMASS",
                    saturation_header_entry="SATURATE", gain_header_entry="GAIN", band_names=None, saturation_level=None, verbose=True):

        """
        Load reduced images (output of cleaning.save_fits_files() or cleaning.master()).

        Parameters
        ----------
        source: string, Path or list
            A fits file, a directory with fits files, or a list of files and/or directories.
        band_header_entry, exposure_header_entry, airmass_header_entry: string
            Header keywords with the filter, the exposure time (s) and the airmass. For master images combined by cleaning.master(),
            EXPTIME and AIRMASS are the means of the combined frames, which is what the photometry needs.
        saturation_header_entry, gain_header_entry: string
            Header keywords with the saturation level (ADU) and the gain (e-/ADU), used to flag saturated stars. ESO instruments, for
            example, keep the gain in keywords such as "ESO DET OUT1 CONAD".
        band_names: dict or None
            Short names for the filters written in the header, e.g. {"B/99": "B", "V/89": "V", "Ic/lwp": "I"}: the band names are used in
            the tables, in the photometric solution and in the color names (B-V), so they must match the columns of the catalogues.
        saturation_level: float or None
            Saturation level in ADU, for headers without a saturation keyword (e.g. 65535 for a 16-bit converter). As with the keyword,
            85% of it (converted to electrons with the gain) is used as the limit, to stay away from the non-linear regime.
        verbose: bool
            If True, print a line per image (for more than 20 images, only a summary).

        Returns
        -------
        images: list
            List of Image objects (attributes data, variance, mask, header, band, exposure_time, airmass, mjd, wcs, ...).
        """

        images = []
        paths = aux.collect_image_paths(source)
        for path in paths:
            image = aux.read_image(path, band_keyword=band_header_entry, exposure_time_keyword=exposure_header_entry,
                                   airmass_keyword=airmass_header_entry, saturation_keyword=saturation_header_entry,
                                   gain_keyword=gain_header_entry)
            image.extra["path"] = str(path)
            if band_names:
                image.band = band_names.get(image.band, image.band)
            if saturation_level is not None:
                in_electrons = str(image.header.get("BUNIT", "")).strip().lower() in ("electron", "electrons", "e-")
                gain = aux.header_value(image.header, gain_header_entry, 1.0) if in_electrons else 1.0
                image.saturation_level = 0.85*float(saturation_level)*gain
            if image.exposure_time is None or image.exposure_time <= 0:
                raise ValueError(f"{path}: no valid exposure time in the header keyword {exposure_header_entry}.")
            if image.airmass is None:
                warnings.warn(f"{path}: no airmass in the header keyword {airmass_header_entry}; the extinction correction will fail.")
            images.append(image)
            if verbose and len(paths) <= 20:
                print(f"{image.name}: band {image.band}, {image.exposure_time:g} s, airmass {image.airmass}, "
                      f"{image.data.shape[1]}x{image.data.shape[0]} px, WCS: {'yes' if image.wcs is not None else 'no'}")
        if verbose and len(paths) > 20:
            print(f"{len(images)} images loaded (bands: {sorted({str(image.band) for image in images})}, "
                  f"{images[0].data.shape[1]}x{images[0].data.shape[0]} px).")
        self.images += images
        return images

    @staticmethod
    def crop_to_detector_window(full_frame, window_header, window_keyword="DETSEC"):

        """
        Cut a full-frame calibration master (and its variance) to the region of interest (window) read out in a science frame.

        Use it when the science frames were read out in a sub-window of the detector (e.g. fast time series for transits), but the master
        bias and flat were made from full frames: the masters are made with cleaning as usual, cropped with this function, and then given to
        cleaning.debias(), cleaning.flatten() and cleaning.compute_variance() of the session that reduces the windowed frames.

        Parameters
        ----------
        full_frame: string, Path, array or tuple
            A master file written by easyspec (read with easyspec.data_io.read_master, so the VAR extension is also cropped), an array, or
            a tuple (data, variance).
        window_header: astropy.io.fits.Header, string or Path
            Header (or file) of one windowed science frame.
        window_keyword: string
            Header keyword with the detector section read out, in the FITS convention '[x1:x2,y1:y2]' (1-based, inclusive). The trimmed
            master must cover the full detector starting at pixel 1 (as done by cleaning.trim() with the overscan removed).

        Returns
        -------
        data, variance: arrays
            The cropped master and its variance (None if not available).
        """

        from astropy.io import fits
        from easyspec import data_io
        if isinstance(window_header, (str, Path)):
            window_header = fits.getheader(window_header)
        if window_keyword not in window_header:
            raise KeyError(f"The keyword {window_keyword} is not in the header of the windowed frame.")
        row_slice, column_slice = aux.parse_section(window_header[window_keyword])
        if isinstance(full_frame, (str, Path)):
            data, variance = data_io.read_master(full_frame)
        elif isinstance(full_frame, tuple):
            data, variance = full_frame
        else:
            data, variance = full_frame, None
        data = np.asarray(data)
        if row_slice.stop > data.shape[0] or column_slice.stop > data.shape[1]:
            raise ValueError(f"The window {window_header[window_keyword]} does not fit in the master ({data.shape[1]}x{data.shape[0]}).")
        cropped_variance = None if variance is None else np.asarray(variance)[row_slice, column_slice]
        return data[row_slice, column_slice], cropped_variance

    # ==================================================================================================================================
    # Detection
    # ==================================================================================================================================

    def find_stars(self, images, detection_threshold=5.0, fwhm="auto", background_box_size=64, minimum_separation=2.5,
                   maximum_number_of_sources=None, plot=False, verbose=True):

        """
        Background estimation, FWHM measurement and star detection.

        The background is computed in two passes (photutils Background2D with sigma-clipped medians in boxes of background_box_size
        pixels; in the second pass the pixels of the sources found in the first are masked). The FWHM is the median of 2D Gaussian fits to
        bright, isolated, unsaturated stars, and the stars are then detected with DAOStarFinder using that FWHM.

        Parameters
        ----------
        images: Image or list of Image
            Images from load_images().
        detection_threshold: float
            Detection threshold, in units of the background RMS.
        fwhm: "auto" or float
            FWHM in pixels. "auto" measures it in each image.
        background_box_size: int
            Size (pixels) of the boxes of the background map. It must be larger than the stars, and smaller than the scale of the
            background variations (e.g. the diffuse light of a galaxy or globular cluster: use ~32 there).
        minimum_separation: float
            Minimum distance between two detections, in units of the FWHM (the default of photutils >= 3.0). Smaller values separate
            close pairs in crowded fields, at the price of spurious detections on the wings of bright stars (psf_photometry() finds the
            close companions in its residual images instead).
        maximum_number_of_sources: int or None
            Keep only the brightest detections.
        plot: bool
            If True, show each image with the detections.
        verbose: bool
            If True, print a line per image.

        Returns
        -------
        sources: QTable or list of QTable
            Tables with id, x, y (0-based pixels), peak and flux_estimate. They are also stored in image.extra["sources"].
        """

        single_image = isinstance(images, aux.Image)
        output = []
        for image in ([images] if single_image else images):
            image.background, image.background_rms = aux.estimate_background(image, box_size=background_box_size)
            source_mask = aux.source_mask_from_threshold(image.data, image.background, image.background_rms)
            image.background, image.background_rms = aux.estimate_background(image, box_size=background_box_size, source_mask=source_mask)
            image.extra["source_mask"] = aux.source_mask_from_threshold(image.data, image.background, image.background_rms)
            image.extra["background_box_size"] = background_box_size
            bright_sources = aux.detect_stars(image, threshold_in_sigmas=max(detection_threshold, 10.0),
                                              fwhm=3.0 if fwhm == "auto" else fwhm, maximum_number_of_sources=500)
            number_of_fwhm_stars = 0
            fwhm_scatter = 0.0
            if fwhm == "auto":
                value, fwhm_scatter, number_of_fwhm_stars = aux.measure_fwhm(image, bright_sources)
                if not np.isfinite(value):
                    raise RuntimeError(f"{image.name}: could not measure the FWHM (no bright isolated stars). Pass fwhm=<value in pixels>.")
                # Second iteration with stamps adapted to the measured FWHM (important for defocused images)
                if value > 6:
                    bright_sources = aux.detect_stars(image, threshold_in_sigmas=max(detection_threshold, 10.0), fwhm=value,
                                                      maximum_number_of_sources=500)
                    value, fwhm_scatter, number_of_fwhm_stars = aux.measure_fwhm(image, bright_sources,
                                                                                 stamp_size=int(2*np.ceil(3*value) + 1))
                image.fwhm = value
            else:
                image.fwhm = float(fwhm)
            sources = aux.detect_stars(image, threshold_in_sigmas=detection_threshold, fwhm=image.fwhm,
                                       maximum_number_of_sources=maximum_number_of_sources, minimum_separation=minimum_separation)
            image.extra["sources"] = sources
            output.append(sources)
            if verbose:
                seeing = image.fwhm*aux.pixel_scale_of(image)
                text = f"FWHM = {image.fwhm:.2f} px" + (f" ({seeing:.2f} arcsec)" if np.isfinite(seeing) else "")
                if number_of_fwhm_stars:
                    text += f", scatter {fwhm_scatter:.2f} px from {number_of_fwhm_stars} stars"
                print(f"{image.name}: {text}; sky = {np.median(image.background):.1f} e-, RMS = {np.median(image.background_rms):.2f} e-; "
                      f"{len(sources)} sources above {detection_threshold} sigma")
            if plot:
                self.plot_image(image, sources=sources)
        return output[0] if single_image else output

    # ==================================================================================================================================
    # Aperture photometry
    # ==================================================================================================================================

    def aperture_photometry(self, images, aperture_radius=1.2, sky_annulus=(4.5, 6.5), correction_radius=4.0,
                            apply_aperture_correction=True, correct_background_noise=True, positions=None, detection_threshold=5.0,
                            fwhm="auto", background_box_size=64, verbose=True):

        """
        Aperture photometry with a local sky and aperture correction.

        For each star, the sky is the sigma-clipped median of the annulus (excluding masked pixels and discs around all other detected
        sources), the flux is the sum in a circular aperture (exact pixel overlap) minus the sky, and the uncertainty combines the variance
        of the pixels (VAR extension: photon, read, bias and flat-field noise) with the uncertainty of the sky level. A small aperture
        maximizes the S/N of faint stars; the light lost outside it is restored with an aperture correction measured on bright, isolated
        stars of the same image (ratio between the fluxes in correction_radius and aperture_radius).

        Parameters
        ----------
        images: Image or list of Image
            Images from load_images(). If find_stars() was not run, it is run here (with detection_threshold, fwhm and
            background_box_size).
        aperture_radius: float
            Radius of the photometry aperture in units of the FWHM of each image (default 1.2 FWHM, close to the S/N optimum).
        sky_annulus: tuple
            Inner and outer radii of the sky annulus, in units of the FWHM. The inner radius should be beyond the correction radius,
            so that the PSF wings do not raise the sky level.
        correction_radius: float
            Radius (in FWHM) of the large aperture used for the aperture correction. At 4 FWHM a Moffat PSF with beta = 3 still misses
            ~0.3% of the light; this is the same in all images with similar PSFs and is absorbed by the zero point.
        apply_aperture_correction: bool
            If True, apply the aperture correction. Its uncertainty (common to all stars of the image) and the intrinsic star-to-star
            scatter of the aperture loss are reported in table.meta ("aperture_correction_magnitude_error", "aperture_correction_scatter").
        correct_background_noise: bool
            If True, the errors are multiplied by the ratio between the measured scatter of sums in empty-sky apertures and the scatter
            predicted by the VAR extension ("background noise factor", >= 1). It accounts for the noise correlation introduced by the
            sub-pixel interpolation of cleaning.align(), which the VAR extension does not include, and for the confusion noise of
            unresolved stars in crowded fields. For images that were not resampled, in uncrowded fields, the factor is ~1.
        positions: array (N, 2) or None
            Pixel positions (0-based x, y) to measure (forced photometry). If None, the detected sources are used.
        detection_threshold, fwhm, background_box_size: see find_stars().
        verbose: bool
            If True, print a summary per image.

        Returns
        -------
        tables: Table or list of Table
            One table per image with id, x, y, (ra, dec), flux, flux_error (e-, aperture corrected), instrumental_magnitude,
            instrumental_magnitude_error (-2.5 log10(flux/exposure time)), sky, sky_standard_deviation, signal_to_noise, flags (see
            photometry.FLAG_BITS). Image metadata (name, band, exposure time, airmass, mjd, fwhm, radii, aperture correction) is in
            table.meta.
        """

        single_image = isinstance(images, aux.Image)
        output = []
        for image in ([images] if single_image else images):
            if "sources" not in image.extra:
                self.find_stars(image, detection_threshold=detection_threshold, fwhm=fwhm, background_box_size=background_box_size,
                                verbose=verbose)
            sources = image.extra["sources"]
            if positions is None:
                star_positions = np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)]
                peaks = np.asarray(sources["peak"], dtype=float)
                identifiers = np.asarray(sources["id"])
            else:
                star_positions = np.atleast_2d(np.asarray(positions, dtype=float))
                identifiers = np.arange(1, len(star_positions) + 1)
                peaks = np.array([self._peak_above_background(image, x, y) for x, y in star_positions])
            if len(star_positions) == 0:
                warnings.warn(f"{image.name}: no sources to measure.")
                output.append(Table())
                continue
            fwhm_in_pixels = image.fwhm
            radius = aperture_radius*fwhm_in_pixels
            inner_radius, outer_radius = sky_annulus[0]*fwhm_in_pixels, sky_annulus[1]*fwhm_in_pixels
            large_radius = correction_radius*fwhm_in_pixels
            if inner_radius < large_radius:
                warnings.warn(f"The inner radius of the sky annulus ({sky_annulus[0]} FWHM) is smaller than the correction radius "
                              f"({correction_radius} FWHM): the PSF wings bias the sky.")
            detected_positions = (np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)] if len(sources)
                                  else star_positions)
            # Pixels kept out of the sky annuli: discs around all detected sources plus the (dilated) pixels above 2 sigma, which also
            # cover the extended wings of bright stars
            excluded_pixels = aux.exclusion_mask(image.data.shape, detected_positions, 1.5*fwhm_in_pixels)
            if image.extra.get("source_mask") is not None:
                excluded_pixels |= image.extra["source_mask"]
            # Brightness of all detected sources in the same units as for the measured stars (background-subtracted sum in the
            # aperture), used for the crowding flag and to select isolated stars for the aperture correction
            detected_fluxes = aux.quick_aperture_fluxes(image, detected_positions, radius)
            noise_factor = 1.0
            if correct_background_noise:
                noise_factor, _ = aux.background_noise_factor(image, radius, source_mask=image.extra.get("source_mask"))
            settings = dict(radius=radius, inner_radius=inner_radius, outer_radius=outer_radius, excluded_pixels=excluded_pixels,
                            detected_positions=detected_positions, detected_fluxes=detected_fluxes, noise_factor=noise_factor,
                            fwhm=fwhm_in_pixels)
            flux, flux_error, sky, sky_standard_deviation, number_of_sky_pixels, signal_to_noise, flags = self._measure_apertures(
                image, star_positions, peaks, **settings)
            correction, correction_error, number_of_correction_stars, correction_scatter = 1.0, 0.0, 0, 0.0
            if apply_aperture_correction:
                # Measured on the detected stars (also for forced photometry, where the requested positions may all be faint)
                if positions is None:
                    correction_inputs = (star_positions, flux, flux_error, flags, sky, sky_standard_deviation, number_of_sky_pixels)
                else:
                    detected_flux, detected_error, detected_sky, detected_sky_std, detected_sky_pixels, _, detected_flags = \
                        self._measure_apertures(image, detected_positions, np.asarray(sources["peak"], dtype=float), **settings)
                    correction_inputs = (detected_positions, detected_flux, detected_error, detected_flags, detected_sky, detected_sky_std,
                                         detected_sky_pixels)
                (correction_positions, correction_flux, correction_flux_error, correction_flags, correction_sky, correction_sky_std,
                 correction_sky_pixels) = correction_inputs
                correction, correction_error, number_of_correction_stars, correction_scatter = aux.aperture_correction(
                    image, correction_positions, correction_flux, correction_flux_error, correction_flags, large_radius, correction_sky,
                    correction_sky_std, correction_sky_pixels, neighbour_positions=detected_positions, neighbour_fluxes=detected_fluxes,
                    own_fluxes=aux.quick_aperture_fluxes(image, correction_positions, radius))
                if number_of_correction_stars == 0:
                    warnings.warn(f"{image.name}: not enough bright isolated stars for the aperture correction; none applied.")
                    correction_error = 0.0
            table = self._photometry_table(image, identifiers, star_positions, flux*correction, flux_error*correction, sky,
                                           sky_standard_deviation, signal_to_noise, flags)
            table.meta.update({"method": "aperture", "aperture_radius": radius, "sky_annulus": (inner_radius, outer_radius),
                               "aperture_correction": correction,
                               "aperture_correction_magnitude_error": (MAGNITUDE_ERROR_FACTOR*correction_error/correction
                                                                       if correction_error else 0.0),
                               "number_of_aperture_correction_stars": number_of_correction_stars,
                               "aperture_correction_scatter": correction_scatter, "background_noise_factor": noise_factor})
            image.extra["photometry"] = table
            output.append(table)
            if verbose:
                print(f"{image.name}: {len(table)} stars ({np.count_nonzero(flags == 0)} unflagged); aperture {radius:.1f} px, annulus "
                      f"{inner_radius:.1f}-{outer_radius:.1f} px; aperture correction {-2.5*np.log10(correction):+.3f} mag from "
                      f"{number_of_correction_stars} stars (scatter {100*correction_scatter:.1f}%); background noise factor "
                      f"{noise_factor:.2f}")
            if noise_factor > 1.5:
                warnings.warn(f"{image.name}: the background fluctuates {noise_factor:.1f}x more than the pixel variance predicts (unresolved "
                              "stars or strongly correlated noise); the errors were scaled accordingly. In crowded fields aperture photometry "
                              "is biased by blends: check the 'crowded' flag (16) and prefer psf_photometry().")
        return output[0] if single_image else output

    @staticmethod
    def _measure_apertures(image, star_positions, peaks, radius, inner_radius, outer_radius, excluded_pixels, detected_positions,
                           detected_fluxes, noise_factor, fwhm):
        """Local sky, aperture flux, error, S/N and flags of the stars at star_positions."""
        row_index = np.clip(np.round(star_positions[:, 1]).astype(int), 0, image.data.shape[0] - 1)
        column_index = np.clip(np.round(star_positions[:, 0]).astype(int), 0, image.data.shape[1] - 1)
        background_at_star = image.background[row_index, column_index]
        sky, sky_standard_deviation, number_of_sky_pixels = aux.local_sky_in_annuli(image, star_positions, inner_radius, outer_radius,
                                                                                    excluded_pixels=excluded_pixels)
        # Fallback (crowded regions where the annulus is almost entirely covered by stars): the background map
        missing = ~np.isfinite(sky)
        if np.any(missing):
            sky[missing] = background_at_star[missing]
            sky_standard_deviation[missing] = image.background_rms[row_index[missing], column_index[missing]]
            number_of_sky_pixels[missing] = int(np.pi*(outer_radius**2 - inner_radius**2))
        flux, flux_error, _ = aux.aperture_fluxes(image, star_positions, radius, sky, sky_standard_deviation, number_of_sky_pixels)
        flux_error = flux_error*noise_factor
        with np.errstate(divide="ignore", invalid="ignore"):
            signal_to_noise = flux/flux_error
        own_fluxes = aux.quick_aperture_fluxes(image, star_positions, radius)
        # Peak values above the background plus the background (for the saturation flag)
        flags = aux.quality_flags(image, star_positions, radius, outer_radius, np.asarray(peaks) + background_at_star,
                                  neighbour_positions=detected_positions, neighbour_fluxes=detected_fluxes, own_fluxes=own_fluxes,
                                  crowding_radius=radius + 1.5*fwhm, signal_to_noise=signal_to_noise)
        return flux, flux_error, sky, sky_standard_deviation, number_of_sky_pixels, signal_to_noise, flags

    @staticmethod
    def _peak_above_background(image, x, y, half_size=2):
        """Maximum pixel value (above the background) around (x, y)."""
        x_center, y_center = int(round(x)), int(round(y))
        box = (image.data - image.background)[max(y_center - half_size, 0):y_center + half_size + 1,
                                              max(x_center - half_size, 0):x_center + half_size + 1]
        return float(np.nanmax(box)) if box.size else np.nan

    @staticmethod
    def _photometry_table(image, identifiers, star_positions, flux, flux_error, sky, sky_standard_deviation, signal_to_noise, flags):
        """Common output table of aperture_photometry() and psf_photometry()."""
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = -2.5*np.log10(flux/image.exposure_time)
            magnitude_error = MAGNITUDE_ERROR_FACTOR*flux_error/flux
        magnitude[~(flux > 0)] = np.nan
        magnitude_error[~(flux > 0)] = np.nan
        table = Table()
        table["id"] = identifiers
        table["x"], table["y"] = star_positions[:, 0], star_positions[:, 1]
        if image.wcs is not None:
            sky_coordinates = image.wcs.celestial.pixel_to_world(star_positions[:, 0], star_positions[:, 1])
            table["ra"], table["dec"] = sky_coordinates.ra.deg, sky_coordinates.dec.deg
        table["flux"], table["flux_error"] = flux, flux_error
        table["instrumental_magnitude"], table["instrumental_magnitude_error"] = magnitude, magnitude_error
        table["sky"], table["sky_standard_deviation"] = sky, sky_standard_deviation
        table["signal_to_noise"] = signal_to_noise
        table["flags"] = flags
        table.meta.update({"image": image.name, "band": image.band, "exposure_time": image.exposure_time, "airmass": image.airmass,
                           "mjd": image.mjd, "object": image.object_name, "fwhm": image.fwhm,
                           "instrument": str(image.header.get("INSTRUME", "")).strip(), "pixel_scale": aux.pixel_scale_of(image),
                           "path": image.extra.get("path", image.name)})
        table.meta["wcs"] = image.wcs
        return table

    # ==================================================================================================================================
    # PSF photometry
    # ==================================================================================================================================

    def psf_photometry(self, images, number_of_psf_stars=30, psf_oversampling=2, fit_size=2.0, grouping_separation=2.0,
                       number_of_iterations=3, residual_detection_threshold=5.0, psf_model_error_fraction=0.05, correction_radius=4.0,
                       detection_threshold=5.0, fwhm="auto", background_box_size=32, plot=True, plot_region=None, image_scale="log",
                       verbose=True):

        """
        PSF ("profile") photometry with an empirical PSF, for crowded fields (globular cluster cores, dense Galactic fields).

        Steps (per image):
            1. background map with small boxes (background_box_size), which follows the diffuse light of clusters and galaxies, FWHM
               and detection (find_stars());
            2. empirical PSF (photutils EPSFBuilder, Anderson & King 2000) from the brightest isolated, unsaturated stars;
            3. simultaneous PSF fit of each group of overlapping stars (photutils PSFPhotometry);
            4. the PSF is rebuilt from its stars with all their fitted neighbours subtracted (in crowded fields the neighbours would
               otherwise contaminate the wings of the PSF); new stars are detected in the image with all fitted stars subtracted (close
               companions, stars in the wings of bright stars) using a noise map that includes a PSF-model uncertainty
               (psf_model_error_fraction x model), and all stars are fitted again; repeated number_of_iterations times. Saturated pixels
               are excluded from the fits (saturated stars are fitted with their wings), and stars that converge to the same position
               are merged;
            5. PSF-to-total correction: ratio between the flux in a large aperture (correction_radius FWHM, with all neighbours
               subtracted) and the PSF flux of bright isolated stars, so that the PSF magnitudes are on the same scale as the
               aperture-corrected magnitudes and can be calibrated with fit_standard_star_solution() / calibrate_magnitudes().

        Parameters
        ----------
        images: Image or list of Image
            Images from load_images().
        number_of_psf_stars: int
            Maximum number of stars used to build the PSF.
        psf_oversampling: int
            Oversampling of the empirical PSF (2 is robust with a few tens of stars; 4 needs many more stars).
        fit_size: float
            Size of the fitting box, in FWHM (the core of the PSF, where the S/N is highest).
        grouping_separation: float
            Stars closer than this (in FWHM) are fitted simultaneously.
        number_of_iterations: int
            Number of fit + residual-detection cycles.
        residual_detection_threshold: float
            Detection threshold (in sigma of the noise map) in the residual images.
        psf_model_error_fraction: float
            Relative uncertainty of the PSF model used in the noise map of the residual images.
        correction_radius: float
            Radius (in FWHM) of the large aperture for the PSF-to-total correction.
        detection_threshold, fwhm, background_box_size: see find_stars(). If find_stars() was already run on an image, its results
            are used.
        plot: bool
            If True (default), show the diagnostic plots of plot_psf_photometry(): the image, the model of all fitted stars and the
            residual image, one after the other.
        plot_region: ((x_start, x_end), (y_start, y_end)) or None
            Region (pixels) shown in the plots, e.g. a zoom on the core of a cluster. None (default) shows the whole image.
        image_scale: "log" or "linear"
            Intensity scale of the plots (default "log").
        verbose: bool
            If True, print a summary per image.

        Returns
        -------
        tables: Table or list of Table
            Same columns as aperture_photometry() (flux and instrumental_magnitude are total, i.e. PSF-corrected; the errors include
            the fit error, scaled by the background noise factor, and a PSF-model error floor given by the star-to-star scatter of the
            PSF-to-total correction), plus
            psf_fit_quality (photutils qfit: sum of absolute residuals / flux), central_fit_quality (cfit), group_size and
            iteration_detected. Image metadata, the PSF-to-total correction and its uncertainty are in table.meta. The PSF model and
            the model image of all stars are stored in image.extra["psf_model"] and image.extra["star_model_image"] (used by
            surface_brightness_profile() to remove the resolved stars).
        """

        single_image = isinstance(images, aux.Image)
        output = []
        for image in ([images] if single_image else images):
            if "sources" not in image.extra or image.extra.get("background_box_size") != background_box_size:
                self.find_stars(image, detection_threshold=detection_threshold, fwhm=fwhm, background_box_size=background_box_size,
                                verbose=verbose)
            sources = image.extra["sources"]
            psf_model, psf_stars, stamp_size = aux_psf.build_empirical_psf(image, sources, number_of_stars=number_of_psf_stars,
                                                                           oversampling=psf_oversampling)
            if verbose:
                print(f"{image.name}: empirical PSF from {len(psf_stars)} stars (stamp {stamp_size} px, oversampling {psf_oversampling})")
            result, model_image, added, psf_model = aux_psf.iterative_psf_photometry(
                image, psf_model, sources, stamp_size, fit_shape_in_fwhm=fit_size, grouping_separation_in_fwhm=grouping_separation,
                number_of_iterations=number_of_iterations, residual_threshold_in_sigmas=residual_detection_threshold,
                psf_model_error_fraction=psf_model_error_fraction, psf_star_table=psf_stars, psf_oversampling=psf_oversampling,
                verbose=verbose)
            star_positions = np.c_[np.asarray(result["x_fit"], dtype=float), np.asarray(result["y_fit"], dtype=float)]
            psf_flux = np.asarray(result["flux_fit"], dtype=float)
            psf_flux_error = np.asarray(result["flux_err"], dtype=float)
            # Errors: the fit errors assume the pixel errors are right; in crowded fields unresolved stars add noise (as for aperture
            # photometry, the background noise factor measures it)
            noise_factor, _ = aux.background_noise_factor(image, image.fwhm, source_mask=image.extra.get("source_mask"))
            psf_flux_error = psf_flux_error*noise_factor
            # Flags: saturation, edge, low S/N and photutils fit flags; "crowded" only for unresolved pairs (< 1 FWHM)
            fwhm_in_pixels = image.fwhm
            peaks = np.array([self._peak_above_background(image, x, y) if np.all(np.isfinite([x, y])) else np.nan
                              for x, y in star_positions])
            with np.errstate(divide="ignore", invalid="ignore"):
                signal_to_noise = psf_flux/psf_flux_error
            row_index = np.clip(np.round(np.nan_to_num(star_positions[:, 1])).astype(int), 0, image.data.shape[0] - 1)
            column_index = np.clip(np.round(np.nan_to_num(star_positions[:, 0])).astype(int), 0, image.data.shape[1] - 1)
            valid = (np.all(np.isfinite(star_positions), axis=1) & (star_positions[:, 0] >= 0) & (star_positions[:, 1] >= 0)
                     & (star_positions[:, 0] <= image.data.shape[1] - 1) & (star_positions[:, 1] <= image.data.shape[0] - 1))
            flags = np.zeros(len(result), dtype=int)
            flags[valid] = aux.quality_flags(image, star_positions[valid], fwhm_in_pixels, fwhm_in_pixels,
                                             peaks[valid] + image.background[row_index[valid], column_index[valid]],
                                             neighbour_positions=star_positions[valid], neighbour_fluxes=np.clip(psf_flux[valid], 0, None),
                                             own_fluxes=np.clip(psf_flux[valid], 0, None), crowding_radius=0.8*fwhm_in_pixels,
                                             signal_to_noise=signal_to_noise[valid])
            flags[~valid] |= 64
            photutils_flags = np.asarray(result["flags"], dtype=int) if "flags" in result.colnames else np.zeros(len(result), int)
            fit_quality = np.asarray(result["qfit"], dtype=float) if "qfit" in result.colnames else np.zeros(len(result))
            flags[(photutils_flags & ~1) != 0] |= 64          # photutils bit 1 (masked pixels in the fit box) is covered by bit 2
            flags[~np.isfinite(fit_quality) | (np.abs(fit_quality) > 0.5) | ~(psf_flux > 0)] |= 64
            # PSF-to-total correction
            correction, correction_error, number_of_correction_stars, correction_scatter = aux_psf.psf_to_total_correction(
                image, psf_model, stamp_size, star_positions, psf_flux, psf_flux_error, flags, model_image,
                correction_radius*fwhm_in_pixels)
            if number_of_correction_stars == 0:
                warnings.warn(f"{image.name}: the PSF-to-total correction could not be measured; none applied.")
                correction_error = 0.0
            # The star-to-star scatter of the PSF-to-total ratios measures the PSF-model error (PSF variations, imperfect empirical PSF):
            # it is an error floor of every star, added in quadrature (it dominates for bright stars)
            psf_flux_error = np.hypot(psf_flux_error, correction_scatter*np.abs(psf_flux))
            with np.errstate(divide="ignore", invalid="ignore"):
                signal_to_noise = psf_flux/psf_flux_error
            table = self._photometry_table(image, np.arange(1, len(result) + 1), star_positions, psf_flux*correction,
                                           psf_flux_error*correction, np.asarray(result["local_bkg"], dtype=float)
                                           if "local_bkg" in result.colnames else np.zeros(len(result)),
                                           np.full(len(result), np.nan), signal_to_noise, flags)
            table["sky"] = image.background[row_index, column_index]
            table["sky_standard_deviation"] = image.background_rms[row_index, column_index]
            table["psf_fit_quality"] = fit_quality
            table["central_fit_quality"] = np.asarray(result["cfit"], dtype=float) if "cfit" in result.colnames else np.nan
            table["group_size"] = np.asarray(result["group_size"]) if "group_size" in result.colnames else 1
            table["iteration_detected"] = np.asarray(result["iteration_detected"])
            table.meta.update({"method": "psf", "aperture_correction": correction,
                               "aperture_correction_magnitude_error": (MAGNITUDE_ERROR_FACTOR*correction_error/correction
                                                                       if correction_error else 0.0),
                               "number_of_aperture_correction_stars": number_of_correction_stars,
                               "aperture_correction_scatter": correction_scatter, "background_noise_factor": noise_factor,
                               "number_of_psf_stars": len(psf_stars), "stars_added_per_iteration": added})
            image.extra["photometry"] = table
            image.extra["psf_model"] = psf_model
            image.extra["psf_stamp_size"] = stamp_size
            image.extra["star_model_image"] = model_image
            output.append(table)
            if verbose:
                print(f"{image.name}: {len(table)} stars ({np.count_nonzero(flags == 0)} unflagged); PSF-to-total correction "
                      f"{-2.5*np.log10(correction):+.3f} mag from {number_of_correction_stars} stars; background noise factor "
                      f"{noise_factor:.2f}")
            if plot:
                self.plot_psf_photometry(image, region=plot_region, image_scale=image_scale)
        return output[0] if single_image else output

    def plot_psf_photometry(self, image, region=None, image_scale="log", show_stars=True, percentile_interval=99.5):

        """
        Diagnostic plots of the PSF photometry of an image (run psf_photometry() first): three large figures, one after the other, with the
        same intensity scale, so that they can be compared directly:

            1. the image (background subtracted), with the fitted stars marked: blue circles are unflagged stars, orange circles are
               flagged ones (saturated, crowded, poor fit, ...);
            2. the model: the PSF of every fitted star, at its fitted position and flux. Crosses mark the stars that were not detected
               at first, but found in the residual images (close companions, stars hidden in the wings of bright stars);
            3. the residual, image - model: what the fit did not explain. A good fit leaves only noise and the light of the stars too faint
               to be detected (and, in a cluster, of its unresolved stars); dark or bright rings around stars indicate a poor PSF model,
               and isolated bright spots are stars that were missed. Saturated stars always leave residuals (their cores cannot be modelled).

        Parameters
        ----------
        image: Image
            Image processed by psf_photometry().
        region: ((x_start, x_end), (y_start, y_end)) or None
            Show only this region (pixels), e.g. a zoom on the core of a cluster. None (default) shows the whole image.
        image_scale: "log" or "linear"
            Intensity scale of the display (default "log").
        show_stars: bool
            Mark the fitted stars on the image and on the model.
        percentile_interval: float
            Percentage of pixels inside the display range.
        """

        if "star_model_image" not in image.extra:
            raise RuntimeError(f"{image.name}: run psf_photometry() on this image first.")
        model_image = image.extra["star_model_image"]
        data = image.data - image.background
        residual = data - model_image
        x_offset, y_offset = 0, 0
        if region is not None:
            (x_start, x_end), (y_start, y_end) = region
            window = (slice(y_start, y_end), slice(x_start, x_end))
            data, model_image, residual = data[window], model_image[window], residual[window]
            x_offset, y_offset = x_start, y_start
        extent = (x_offset - 0.5, x_offset + data.shape[1] - 0.5, y_offset - 0.5, y_offset + data.shape[0] - 0.5)
        normalization = _image_normalization(data, image_scale, percentile_interval)
        table = image.extra.get("photometry")
        stars_inside = None
        if show_stars and table is not None:
            x, y = np.asarray(table["x"], dtype=float), np.asarray(table["y"], dtype=float)
            stars_inside = (x > extent[0]) & (x < extent[1]) & (y > extent[2]) & (y < extent[3])
        panels = ((data, f"1/3  Image $-$ background ({image.name})"),
                  (model_image, "2/3  Model: PSF of every fitted star"),
                  (residual, r"3/3  Residual: image $-$ model"))
        for index, (panel, title) in enumerate(panels):
            with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
                figure, axis = plt.subplots(figsize=(10, 9))
                shown = axis.imshow(panel, origin="lower", cmap="magma", norm=normalization, extent=extent, interpolation="nearest")
                color_bar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
                color_bar.set_label(r"e$^-$ (" + image_scale + " scale)")
                axis.set_title(title, fontsize=13)
                axis.set_xlabel("x (pixels)")
                axis.set_ylabel("y (pixels)")
                axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.4, color="white")
                axis.tick_params(direction="in", top=True, right=True, color="0.6")
                if stars_inside is not None and index < 2:
                    marker_size = (1.6*image.fwhm)**2*(1 if region is not None else 0.5)
                    if index == 0:
                        unflagged = stars_inside & (np.asarray(table["flags"]) == 0)
                        flagged = stars_inside & ~unflagged
                        axis.scatter(x[unflagged], y[unflagged], s=marker_size, facecolors="none", edgecolors="#4FC3F7", linewidths=0.7,
                                     label=f"unflagged ({np.count_nonzero(unflagged)})")
                        axis.scatter(x[flagged], y[flagged], s=marker_size, facecolors="none", edgecolors="#FFB74D", linewidths=0.7,
                                     label=f"flagged ({np.count_nonzero(flagged)})")
                        axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
                    elif "iteration_detected" in table.colnames:
                        found_later = stars_inside & (np.asarray(table["iteration_detected"]) > 0)
                        axis.scatter(x[found_later], y[found_later], s=0.4*marker_size, marker="x", color="#4FC3F7", linewidths=0.7,
                                     label=f"found in the residual images ({np.count_nonzero(found_later)})")
                        axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
                plt.tight_layout()
                plt.show()

    # ==================================================================================================================================
    # Matching
    # ==================================================================================================================================

    def cross_match_with_catalogue(self, photometry_table, catalogue, band=None, maximum_separation=None, search_radius=30.0,
                                   verbose=True):

        """
        Cross-match a photometry table of an image with a WCS with a catalogue of standard (or reference) stars with coordinates.

        The catalogue positions are projected onto the image with its WCS, and the (usually small) offset between the WCS and the true
        positions is found by a vote on the pairwise differences between the brightest catalogue stars and the brightest detections.
        Images without a WCS cannot be matched to a catalogue of coordinates: for standard stars, give their positions by hand with
        fit_standard_star_solution(..., standard_stars=...).

        Parameters
        ----------
        photometry_table: Table
            Output of aperture_photometry() or psf_photometry() for an image with WCS.
        catalogue: string, Path or Table
            Catalogue file (csv), table, or name of a catalogue distributed with easyspec (e.g. "landolt"), with name, ra, dec (deg) and
            magnitudes (magnitudes can be given per band and/or as V plus color indices such as "B-V").
        band: string
            Band of the image (default: photometry_table.meta["band"]); used to sort the catalogue by brightness for the offset vote.
        maximum_separation: float
            Maximum distance (pixels) between a detection and a catalogue star after the offset correction. Default: 1 FWHM.
        search_radius: float
            Maximum WCS offset (pixels) considered by the vote.
        verbose: bool
            If True, print the result.

        Returns
        -------
        matched: Table
            Rows of the photometry table that have a catalogue counterpart, with the catalogue columns appended (prefix "catalogue_" for
            name, ra and dec) and the matching distance (column "match_distance", pixels).
        """

        catalogue = aux.read_catalogue(catalogue)
        wcs = photometry_table.meta.get("wcs")
        if wcs is None:
            raise ValueError(f"{photometry_table.meta.get('image')}: the image has no WCS, so the stars of a catalogue cannot be located on "
                             "it. For standard stars, give their positions (x, y) by hand: fit_standard_star_solution(..., "
                             "standard_stars=...).")
        band = band or photometry_table.meta.get("band")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_catalogue, y_catalogue = wcs.celestial.world_to_pixel(SkyCoord(catalogue["ra"], catalogue["dec"], unit="deg"))
        catalogue_positions = np.c_[np.asarray(x_catalogue, dtype=float), np.asarray(y_catalogue, dtype=float)]
        # Only the catalogue stars that fall on the image (within the search radius of the detections): a large catalogue (e.g. all the
        # Landolt fields) has many brighter stars elsewhere in the sky, which would otherwise take the place of the stars of this field
        # in the vote below
        detected_x, detected_y = np.asarray(photometry_table["x"], dtype=float), np.asarray(photometry_table["y"], dtype=float)
        on_image = (np.all(np.isfinite(catalogue_positions), axis=1)
                    & (catalogue_positions[:, 0] > np.nanmin(detected_x) - search_radius)
                    & (catalogue_positions[:, 0] < np.nanmax(detected_x) + search_radius)
                    & (catalogue_positions[:, 1] > np.nanmin(detected_y) - search_radius)
                    & (catalogue_positions[:, 1] < np.nanmax(detected_y) + search_radius))
        if not np.any(on_image):
            raise ValueError(f"{photometry_table.meta.get('image')}: no star of the catalogue falls on the image according to its WCS. "
                             "Check that the field is in the catalogue (photometry.standard_star_catalogue(name, field=...)) and that the "
                             "WCS is right; otherwise give the standards by hand (standard_stars=...).")
        catalogue = catalogue[on_image]
        catalogue_positions = catalogue_positions[on_image]
        catalogue_order = np.argsort(np.asarray(catalogue[band], dtype=float)) if band in catalogue.colnames else np.arange(len(catalogue))
        detection_order = np.argsort(-np.nan_to_num(np.asarray(photometry_table["flux"], dtype=float), nan=-np.inf))
        detection_positions = np.c_[photometry_table["x"], photometry_table["y"]]
        fwhm = photometry_table.meta.get("fwhm", 3.0)
        maximum_separation = maximum_separation or max(fwhm, 2.0)
        # The offset of the WCS: a vote between all pairs (catalogue star, detection). Two catalogue stars are enough (two stars that
        # agree on the same offset are very unlikely by chance); with a single one, the WCS is used as it is
        dx, dy, number_voted = aux.estimate_translation(detection_positions[detection_order], catalogue_positions[catalogue_order],
                                                        maximum_offset=search_radius, number_of_bright_sources=100,
                                                        bin_size=max(1.0, fwhm/2), minimum_sources=2)
        if number_voted == 0:
            warnings.warn(f"{photometry_table.meta.get('image')}: the offset of the WCS could not be measured ({len(catalogue)} catalogue "
                          f"star(s) on the image); the WCS is used as it is. If the standards are not found, increase search_radius or "
                          "give them by hand (standard_stars=...).")
        detection_index, catalogue_index, distance = aux.match_positions(detection_positions, catalogue_positions,
                                                                         maximum_separation=maximum_separation, offset=(dx, dy))
        matched = photometry_table[detection_index]
        catalogue_rows = catalogue[catalogue_index]
        for name in ("name", "ra", "dec"):
            catalogue_rows.rename_column(name, "catalogue_" + name)
        matched = hstack([matched, catalogue_rows], join_type="exact", metadata_conflicts="silent")
        matched["match_distance"] = distance
        matched.meta = dict(photometry_table.meta)
        if verbose:
            print(f"{photometry_table.meta.get('image')}: {len(matched)} of the {len(catalogue)} catalogue stars on the image matched (WCS offset "
                  f"dx = {dx:+.1f}, dy = {dy:+.1f} px)")
        return matched

    def cross_match_tables(self, reference_table, other_table, maximum_separation=None, search_radius=None, verbose=True):

        """
        Cross-match two photometry tables of the same field (e.g. the B and V images of a cluster), by pixel positions.

        The images do not need to be aligned with each other: the translation between them is found by a vote on the pairwise
        differences of the brightest stars (robust to stars detected in only one of the images). This is needed because master images
        combined by cleaning.master() have no WCS.

        Parameters
        ----------
        reference_table, other_table: Table
            Outputs of aperture_photometry() or psf_photometry() (they need the columns x, y and flux).
        maximum_separation: float
            Maximum distance (pixels) after the offset correction. Default: 1 FWHM (the largest of both images).
        search_radius: float or None
            Maximum offset (pixels) considered. None means no limit.
        verbose: bool
            If True, print the offset and number of matches.

        Returns
        -------
        reference_index, other_index: arrays
            Indices of the matched rows in both tables.
        offset: tuple
            (dx, dy) such that other position + offset = reference position.
        """

        reference_positions = np.c_[reference_table["x"], reference_table["y"]]
        other_positions = np.c_[other_table["x"], other_table["y"]]
        reference_order = np.argsort(-np.nan_to_num(np.asarray(reference_table["flux"], dtype=float), nan=-np.inf))
        other_order = np.argsort(-np.nan_to_num(np.asarray(other_table["flux"], dtype=float), nan=-np.inf))
        fwhm = max(reference_table.meta.get("fwhm", 3.0), other_table.meta.get("fwhm", 3.0))
        dx, dy, _ = aux.estimate_translation(reference_positions[reference_order], other_positions[other_order],
                                             maximum_offset=search_radius, number_of_bright_sources=200, bin_size=max(1.0, fwhm/2))
        maximum_separation = maximum_separation or max(fwhm, 2.0)
        reference_index, other_index, _ = aux.match_positions(reference_positions, other_positions, maximum_separation=maximum_separation,
                                                              offset=(dx, dy))
        # Refine with all matches (sub-pixel)
        if len(reference_index) >= 5:
            shift = np.median(reference_positions[reference_index] - other_positions[other_index], axis=0)
            reference_index, other_index, _ = aux.match_positions(reference_positions, other_positions,
                                                                  maximum_separation=maximum_separation, offset=shift)
            dx, dy = shift
        if verbose:
            print(f"{other_table.meta.get('image')} -> {reference_table.meta.get('image')}: offset dx = {dx:+.2f}, dy = {dy:+.2f} px; "
                  f"{len(reference_index)} stars matched")
        return reference_index, other_index, (float(dx), float(dy))

    # ==================================================================================================================================
    # Calibration with standard stars
    # ==================================================================================================================================

    @staticmethod
    def available_standard_star_catalogues():

        """
        Names of the catalogues of standard stars distributed with easyspec, which can be given by name to fit_standard_star_solution() and
        standard_star_catalogue():
            "landolt_1992", "landolt_2009", "landolt_2013": UBVRI standards of Landolt ("landolt" = 2009 + 2013);
            "clem_landolt_2013", "clem_landolt_2016": faint UBVRI standards (V down to ~22) in the Landolt fields ("clem_landolt" = both),
            for large telescopes, on which the Landolt stars saturate;
        The csv files are created once by standard_star_catalogues/build_standard_star_catalogues.py.
        """

        return aux.available_standard_star_catalogues()

    @staticmethod
    def standard_star_catalogue(name="landolt", field=None):

        """
        A catalogue of standard stars distributed with easyspec, optionally only the stars of one field.

        Parameters
        ----------
        name: string
            Name of the catalogue (see available_standard_star_catalogues()), e.g. "landolt" (Landolt 2009 + 2013, UBVRI) or
            "clem_landolt" (faint UBVRI standards).
        field: string or None
            Name of a field (e.g. "SA98", "PG0918", "RU149"): only its stars are returned. Spaces, dashes and the prefix "SA" do not matter.

        Returns
        -------
        catalogue: Table
            Columns name, field, ra, dec (deg, J2000), the magnitudes and colors of the catalogue and their errors (prefix "e_").
        """

        catalogue = aux.read_catalogue(name)
        if field is not None:
            key = aux.normalized_star_name(field)
            fields = catalogue["field"] if "field" in catalogue.colnames else catalogue["name"]
            selected = np.array([aux.normalized_star_name(value) == key or aux.normalized_star_name(value).startswith(key + " ")
                                 for value in fields])
            catalogue = catalogue[selected]
            if len(catalogue) == 0:
                raise ValueError(f"No star of the field '{field}' in the catalogue '{name}'.")
        return catalogue

    @staticmethod
    def _find_photometry_table(photometry_tables, reference):
        """The photometry table of 'reference': a table, an Image, or a file name / end of a path (e.g. "standards_V/standard_000.fits")."""
        if isinstance(reference, Table):
            for table in photometry_tables:
                if table is reference:
                    return table
            reference = reference.meta.get("path", reference.meta.get("image"))
        if isinstance(reference, aux.Image):
            reference = reference.extra.get("path", reference.name)
        wanted = Path(str(reference)).as_posix()
        candidates = [table for table in photometry_tables
                      if Path(str(table.meta.get("path", table.meta.get("image")))).as_posix().endswith(wanted)
                      or Path(str(table.meta.get("path", ""))).resolve().as_posix() == Path(wanted).resolve().as_posix()]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"No photometry table of the image '{reference}'.")
        raise ValueError(f"'{reference}' matches several images ({', '.join(str(table.meta.get('path')) for table in candidates)}): give "
                         "more of its path, e.g. 'standards_V/standard_star_000.fits'.")

    def _standard_stars_from_positions(self, photometry_tables, standard_stars, standard_stars_image, catalogue, needed_bands,
                                       position_tolerance, minimum_common_stars, verbose):

        """
        The standard stars given by hand (x, y in one frame, and their magnitudes or names) in every frame: for each frame, the shift from
        the frame where the positions were measured is found by matching the stars of both frames (cross_match_tables()), and each standard
        is the detected star closest to its shifted position. Returns {id(table): (table rows, standards rows)}.
        """

        stars = aux.read_catalogue(standard_stars, require_coordinates=False)
        for column in ("x", "y"):
            if column not in stars.colnames:
                raise ValueError(f"standard_stars needs the columns x and y (pixels in the frame where they were measured); it has "
                                 f"{stars.colnames}.")
        missing_bands = [band for band in needed_bands if band not in stars.colnames]
        if missing_bands:
            # magnitudes from a catalogue, by name
            if catalogue is None or "name" not in stars.colnames or all(str(name).startswith("star_") for name in stars["name"]):
                raise ValueError(f"standard_stars has no magnitudes in the bands {missing_bands}: give them (e.g. columns V, B-V, V-I), or "
                                 "give the names of the stars (column 'name') and a catalogue with their magnitudes (e.g. "
                                 "catalogue='landolt').")
            catalogue_table = aux.read_catalogue(catalogue, require_coordinates=False)
            rows_by_name = {aux.normalized_star_name(name): index for index, name in enumerate(catalogue_table["name"])}
            not_found = [str(name) for name in stars["name"] if aux.normalized_star_name(name) not in rows_by_name]
            if not_found:
                raise ValueError(f"Stars not found in the catalogue: {', '.join(not_found)}.")
            indices = [rows_by_name[aux.normalized_star_name(name)] for name in stars["name"]]
            for band in missing_bands:
                if band not in catalogue_table.colnames:
                    raise ValueError(f"The catalogue has no magnitudes in the band {band}.")
                stars[band] = np.asarray(catalogue_table[band], dtype=float)[indices]
        # frames where the positions were measured
        if "image" in stars.colnames:
            references = [self._find_photometry_table(photometry_tables, value) for value in stars["image"]]
        else:
            reference = self._find_photometry_table(photometry_tables, standard_stars_image) if standard_stars_image is not None \
                else photometry_tables[0]
            references = [reference]*len(stars)
        reference_tables = []
        for table in references:
            if not any(table is other for other in reference_tables):
                reference_tables.append(table)
        if verbose:
            print("Standard stars given by hand: " + ", ".join(f"{np.count_nonzero([table is reference for table in references])} measured "
                                                           f"on {reference.meta.get('path')}" for reference in reference_tables))
        found = {}
        for table in photometry_tables:
            # shift between the frame where the positions were measured and this frame
            best = None
            for reference in reference_tables:
                if reference is table:
                    best = (reference, (0.0, 0.0), np.inf)
                    break
                reference_index, _, offset = self.cross_match_tables(reference, table, verbose=False)
                if best is None or len(reference_index) > best[2]:
                    best = (reference, offset, len(reference_index))
            reference, offset, number_in_common = best
            if number_in_common < minimum_common_stars:
                warnings.warn(f"{table.meta.get('path')}: only {number_in_common} stars in common with the frame where the standards were "
                              "measured (another field?): frame skipped. Give the positions of its standards (column 'image').")
                continue
            selected = np.array([other is reference for other in references])
            # cross_match_tables: position in this frame + offset = position in the reference frame
            expected = np.c_[np.asarray(stars["x"], dtype=float)[selected] - offset[0],
                             np.asarray(stars["y"], dtype=float)[selected] - offset[1]]
            detections = np.c_[np.asarray(table["x"], dtype=float), np.asarray(table["y"], dtype=float)]
            tolerance = position_tolerance or max(1.5*(table.meta.get("fwhm") or 3.0), 5.0)
            detection_index, star_index, distance = aux.match_positions(detections, expected, maximum_separation=tolerance)
            standards_here = stars[np.flatnonzero(selected)[star_index]]
            found[id(table)] = (table[detection_index], standards_here)
            if verbose:
                shift = "" if reference is table else f" (shift from {reference.meta.get('path')}: dx = {-offset[0]:+.1f}, dy = {-offset[1]:+.1f} px)"
                print(f"   {table.meta.get('path')} ({table.meta.get('band')}): {len(detection_index)} of {np.count_nonzero(selected)} "
                      f"standards found{shift}")
        return found

    def fit_standard_star_solution(self, photometry_tables, catalogue="landolt", standard_stars=None, standard_stars_image=None,
                                   bands=None, color_index=("B", "V"), fit_extinction=True, fit_color_term=True, fixed_extinction=None,
                                   fixed_color_term=None, accepted_flags=0, clip_sigma=3.0, position_tolerance=None, minimum_common_stars=5,
                                   search_radius=30.0, plot=True, verbose=True):

        """
        Photometric solution (zero point, extinction coefficient and color term of each band) from standard stars.

        For each band, the transformation

            M = m_inst + ZP - k X - c (M1 - M2),      m_inst = -2.5 log10(flux [e-] / exposure time [s]),

        is fitted to all measurements of the standards in all frames of that band (weighted least squares with iterative sigma
        clipping), where M is the standard magnitude, X the airmass and M1 - M2 the standard color index given by color_index. A reliable
        extinction coefficient needs standards observed over a range of airmasses (ideally Delta X > 0.5); otherwise fix it with
        fit_extinction=False and fixed_extinction={band: value} (e.g. the site's mean extinction).

        Where are the standards in each frame?
            - frames with a WCS (astrometric solution in the header): the stars of the catalogue are projected on the image with it;
            - frames without a WCS: the standards must be given by hand with standard_stars: their pixel positions (x, y) in one frame
              (e.g. read in DS9, minus 1 because DS9 counts pixels from 1, or in plot_image()) and either their magnitudes or their names
              in the catalogue. The other frames of the same field are shifted with respect to that one (dithering, different filters):
              the shift is found by matching the stars of both frames, so the positions are given only once per field. When given,
              standard_stars is used for all frames, with or without WCS.

        Parameters
        ----------
        photometry_tables: list of Table
            Outputs of aperture_photometry() for the standard-star frames (any mixture of bands; the band is read from table.meta).
        catalogue: string, Path or Table
            Catalogue of the standards: the name of a catalogue distributed with easyspec ("landolt" (default), "clem_landolt" (faint
            standards); see available_standard_star_catalogues()), a csv file or a table with name, ra, dec in
            deg and magnitudes (per band and/or V plus color indices such as "B-V"). Used for frames with WCS, and to find the magnitudes of
            the standards given by name in standard_stars. Not needed if standard_stars has the magnitudes.
        standard_stars: Table, csv file, list of dicts or None
            Standards identified by hand: columns x, y (pixels, 0-based) and either magnitudes (e.g. V, B-V, V-I, or B, V, I) or name (as
            in the catalogue, e.g. "98 650"). Optional column "image": the frame where each position was measured (file name or end of its
            path); otherwise all positions refer to standard_stars_image.
        standard_stars_image: string, Path, Image, Table or None
            Frame where the positions of standard_stars were measured (file name or end of its path, e.g.
            "standards_V_89/standard_star_000.fits"). Default: the first photometry table.
        bands: list or None
            Bands to calibrate (e.g. ("B", "V")); the tables of other bands are ignored. Default: all bands of the tables.
        color_index: tuple
            Bands (M1, M2) of the color index used in the color term, e.g. ("B", "V") for B - V.
        fit_extinction, fit_color_term: bool or dict
            If False, the extinction (color) coefficient is not fitted but fixed to the value in fixed_extinction (fixed_color_term). A
            dict {band: bool} sets this per band.
        fixed_extinction, fixed_color_term: dict or None
            Fixed coefficients {band: value}, used when they are not fitted (default 0).
        accepted_flags: int
            Measurements with (flags & ~accepted_flags) != 0 are not used. With the default (0) only unflagged stars enter the fit.
        clip_sigma: float
            Sigma-clipping threshold.
        position_tolerance: float or None
            Maximum distance (pixels) between a position given in standard_stars and the detected star. Default: the larger of 1.5 FWHM
            and 5 pixels.
        minimum_common_stars: int
            Minimum number of stars in common between a frame and the frame where the standards were measured (otherwise the frame is
            skipped: it is probably another field).
        search_radius: float
            For frames with WCS: largest error of the WCS (pixels) that is corrected by matching the catalogue to the detected stars
            (see cross_match_with_catalogue()). Increase it if the WCS comes only from the telescope pointing and few standards are
            found.
        plot: bool
            If True, plot the residuals against airmass, color and magnitude.
        verbose: bool
            If True, print the solution.

        Returns
        -------
        solution: dict
            {band: fit result}, each a dict with "coefficients" {"ZP", "k", "c"}, "errors", "covariance", "rms", "reduced_chi2",
            "number_used", "number_total", and "data" (table with the measurements). The color index is in solution["color_index"].
            The solution is also stored in self.standard_star_solution.
        """

        first_band, second_band = color_index
        if bands is not None:
            photometry_tables = [table for table in photometry_tables if table.meta.get("band") in bands]
            if not photometry_tables:
                raise ValueError(f"No photometry table in the bands {list(bands)}.")
        table_bands = []
        for table in photometry_tables:
            if table.meta.get("band") not in table_bands:
                table_bands.append(table.meta.get("band"))
        needed_bands = list(dict.fromkeys([first_band, second_band] + table_bands))

        if standard_stars is not None:
            found = self._standard_stars_from_positions(photometry_tables, standard_stars, standard_stars_image, catalogue, needed_bands,
                                                        position_tolerance, minimum_common_stars, verbose)
        else:
            without_wcs = [str(table.meta.get("path")) for table in photometry_tables if table.meta.get("wcs") is None]
            if without_wcs:
                raise ValueError("These frames have no WCS, so the standards of the catalogue cannot be located on them: "
                                 f"{', '.join(without_wcs)}. Give the standards by hand with standard_stars (their x, y in one frame and "
                                 "their magnitudes or names).")
            catalogue_table = aux.read_catalogue(catalogue)
            for band in (first_band, second_band):
                if band not in catalogue_table.colnames:
                    raise ValueError(f"The catalogue has no magnitude for the band {band}; columns: {catalogue_table.colnames}")
            found = {}
            for table in photometry_tables:
                matched = self.cross_match_with_catalogue(table, catalogue_table, search_radius=search_radius, verbose=verbose)
                found[id(table)] = (matched, matched)

        solution = {"color_index": color_index}
        for band in table_bands:
            rows = []
            for table in photometry_tables:
                if table.meta.get("band") != band or id(table) not in found:
                    continue
                measured, standards = found[id(table)]
                if band not in standards.colnames:
                    warnings.warn(f"No standard magnitudes in the band {band}: skipped.")
                    break
                names = standards["catalogue_name"] if "catalogue_name" in standards.colnames else standards["name"]
                instrumental = np.asarray(measured["instrumental_magnitude"], dtype=float)
                usable = ((np.asarray(measured["flags"]) & ~accepted_flags) == 0) & np.isfinite(instrumental)
                for index in np.flatnonzero(usable):
                    rows.append((str(names[index]), table.meta["image"], float(standards[band][index]),
                                 float(standards[first_band][index]) - float(standards[second_band][index]), float(instrumental[index]),
                                 float(measured["instrumental_magnitude_error"][index]), float(table.meta["airmass"]),
                                 float(table.meta.get("aperture_correction_magnitude_error", 0.0))))
            if not rows:
                warnings.warn(f"No usable standard-star measurements in the band {band}.")
                continue
            data = Table(rows=rows, names=("name", "image", "standard_magnitude", "color", "instrumental_magnitude",
                                           "instrumental_magnitude_error", "airmass", "aperture_correction_error"))
            # The aperture-correction uncertainty is common to all stars of a frame; added in quadrature as a (conservative) error term
            errors = np.hypot(np.asarray(data["instrumental_magnitude_error"]), np.asarray(data["aperture_correction_error"]))
            fit_extinction_here = fit_extinction.get(band, True) if isinstance(fit_extinction, dict) else fit_extinction
            fit_color_here = fit_color_term.get(band, True) if isinstance(fit_color_term, dict) else fit_color_term
            airmasses = np.asarray(data["airmass"])
            if fit_extinction_here and np.ptp(airmasses) < 0.2:
                warnings.warn(f"Band {band}: the standards span only Delta X = {np.ptp(airmasses):.2f} in airmass; the extinction coefficient "
                              "is poorly constrained. Consider fit_extinction=False with a known extinction.")
            if fit_color_here and np.ptp(np.asarray(data["color"])) < 0.3:
                warnings.warn(f"Band {band}: the standards span only {np.ptp(np.asarray(data['color'])):.2f} mag in color; the color term "
                              "is poorly constrained.")
            result = aux.fit_photometric_transformation(data["standard_magnitude"], data["instrumental_magnitude"], errors, airmasses,
                                                        data["color"], fit_extinction=fit_extinction_here, fit_color_term=fit_color_here,
                                                        fixed_extinction=(fixed_extinction or {}).get(band, 0.0),
                                                        fixed_color_term=(fixed_color_term or {}).get(band, 0.0), clip_sigma=clip_sigma)
            data["residual"] = result.pop("residuals")
            data["used"] = result.pop("used")
            result["data"] = data
            solution[band] = result
            if verbose:
                values, uncertainties = result["coefficients"], result["errors"]
                print(f"Band {band}: ZP = {values['ZP']:.4f} +/- {uncertainties['ZP']:.4f}, k = {values['k']:.4f} +/- "
                      f"{uncertainties['k']:.4f}{'' if fit_extinction_here else ' (fixed)'}, c = {values['c']:+.4f} +/- "
                      f"{uncertainties['c']:.4f}{'' if fit_color_here else ' (fixed)'} [color {first_band}-{second_band}]; rms = "
                      f"{result['rms']:.4f} mag, reduced chi2 = {result['reduced_chi2']:.2f}, {result['number_used']}/"
                      f"{result['number_total']} measurements used")
        if plot:
            self.plot_standard_star_solution(solution)
        self.standard_star_solution = solution
        return solution

    def plot_standard_star_solution(self, solution=None):
        """Residuals of the standard-star solution against airmass, color and standard magnitude (one row per band)."""
        solution = solution or self.standard_star_solution
        bands = [band for band in solution if band != "color_index"]
        if not bands:
            return
        color_name = "-".join(solution["color_index"])
        figure, axes = plt.subplots(len(bands), 3, figsize=(13, 3.2*len(bands)), squeeze=False, sharey="row")
        for row, band in zip(axes, bands):
            data = solution[band]["data"]
            used = np.asarray(data["used"])
            for axis, column, label in zip(row, ("airmass", "color", "standard_magnitude"), ("Airmass", color_name, band)):
                axis.errorbar(np.asarray(data[column])[used], np.asarray(data["residual"])[used],
                              yerr=np.asarray(data["instrumental_magnitude_error"])[used], fmt="o", ms=4, color="C0", alpha=0.8,
                              label="used")
                if np.any(~used):
                    axis.plot(np.asarray(data[column])[~used], np.asarray(data["residual"])[~used], "x", color="C3", label="rejected")
                axis.axhline(0, color="k", lw=0.8)
                axis.set_xlabel(label)
            row[0].set_ylabel(f"{band}: residual (mag)")
            values = solution[band]["coefficients"]
            row[0].set_title(f"ZP = {values['ZP']:.3f}, k = {values['k']:.3f}, c = {values['c']:+.3f}, rms = {solution[band]['rms']:.3f}",
                             fontsize=10, loc="left")
            row[2].invert_xaxis()
            row[0].legend(fontsize=8)
        figure.tight_layout()
        plt.show()

    # ==================================================================================================================================
    # Calibration of the science images
    # ==================================================================================================================================

    def calibrate_magnitudes(self, photometry_tables, solution=None, maximum_separation=None, color_index_values=None, verbose=True):

        """
        Calibrated magnitudes of the stars of science images.

        With two bands (the bands of solution["color_index"], e.g. B and V), the stars are cross-matched between the images and the two
        transformation equations are solved together for the magnitudes and the color (exact solution, no iteration). With a single
        band, the color term needs the color of the stars: pass color_index_values=<value or array>, otherwise it is neglected (color
        = 0, which gives an error of c x color).

        Parameters
        ----------
        photometry_tables: Table or list of Table
            Outputs of aperture_photometry() or psf_photometry() for the science images: one image per band (e.g. the master images of
            cleaning.master()).
        solution: dict
            Output of fit_standard_star_solution() (default: the last one computed).
        maximum_separation: float
            Maximum distance (pixels) when matching the two bands (default: 1 FWHM).
        color_index_values: float or array
            Color index of the stars, for single-band calibration.
        verbose: bool
            If True, print a summary.

        Returns
        -------
        catalogue: Table
            For two bands: x, y (and ra, dec, if available) in the image of the first band, the magnitudes of both bands and the color,
            with their statistical errors (columns <band>_error, <color>_error), the calibration (systematic) uncertainty in the meta
            ("calibration_error"), and the flags of each band. For one band: the magnitude and its error.
        """

        solution = solution or self.standard_star_solution
        if solution is None:
            raise ValueError("No photometric solution: run fit_standard_star_solution() first.")
        photometry_tables = [photometry_tables] if isinstance(photometry_tables, Table) else list(photometry_tables)
        first_band, second_band = solution["color_index"]
        tables_by_band = {table.meta.get("band"): table for table in photometry_tables}
        if len(tables_by_band) < len(photometry_tables):
            raise ValueError("calibrate_magnitudes() takes one image per band; combine the frames of each band with cleaning.master() first.")

        if first_band in tables_by_band and second_band in tables_by_band and first_band in solution and second_band in solution:
            first_table, second_table = tables_by_band[first_band], tables_by_band[second_band]
            first_index, second_index, offset = self.cross_match_tables(first_table, second_table, maximum_separation=maximum_separation,
                                                                        verbose=verbose)
            instrumental_1 = np.asarray(first_table["instrumental_magnitude"], dtype=float)[first_index]
            instrumental_2 = np.asarray(second_table["instrumental_magnitude"], dtype=float)[second_index]
            error_1 = np.asarray(first_table["instrumental_magnitude_error"], dtype=float)[first_index]
            error_2 = np.asarray(second_table["instrumental_magnitude_error"], dtype=float)[second_index]
            airmass_1, airmass_2 = float(first_table.meta["airmass"]), float(second_table.meta["airmass"])
            (magnitude_1, magnitude_2, color, magnitude_1_error, magnitude_2_error, color_error,
             calibration_errors) = aux.solve_two_band_magnitudes(instrumental_1, instrumental_2, airmass_1, airmass_2,
                                                                 solution[first_band], solution[second_band], error_1, error_2)
            color_name = f"{first_band}-{second_band}"
            catalogue = Table()
            catalogue["id"] = np.arange(1, len(first_index) + 1)
            catalogue["x"], catalogue["y"] = np.asarray(first_table["x"])[first_index], np.asarray(first_table["y"])[first_index]
            if "ra" in first_table.colnames:
                catalogue["ra"] = np.asarray(first_table["ra"])[first_index]
                catalogue["dec"] = np.asarray(first_table["dec"])[first_index]
            catalogue[first_band], catalogue[first_band + "_error"] = magnitude_1, magnitude_1_error
            catalogue[second_band], catalogue[second_band + "_error"] = magnitude_2, magnitude_2_error
            catalogue[color_name], catalogue[color_name + "_error"] = color, color_error
            catalogue[f"flags_{first_band}"] = np.asarray(first_table["flags"])[first_index]
            catalogue[f"flags_{second_band}"] = np.asarray(second_table["flags"])[second_index]
            catalogue["flags"] = catalogue[f"flags_{first_band}"] | catalogue[f"flags_{second_band}"]
            # Aperture (or PSF-to-total) corrections of the science images are a common (systematic) error, added to the calibration error
            correction_error = {band: tables_by_band[band].meta.get("aperture_correction_magnitude_error", 0.0)
                                for band in (first_band, second_band)}
            catalogue.meta.update({"bands": (first_band, second_band), "offset_" + second_band: offset,
                                   "object": first_table.meta.get("object"),
                                   "method": first_table.meta.get("method", "aperture"),
                                   "calibration_error": {first_band: float(np.hypot(calibration_errors[0], correction_error[first_band])),
                                                         second_band: float(np.hypot(calibration_errors[1], correction_error[second_band])),
                                                         color_name: float(np.hypot(calibration_errors[2],
                                                                                    np.hypot(correction_error[first_band],
                                                                                             correction_error[second_band])))}})
            catalogue.sort(second_band)
            catalogue["id"] = np.arange(1, len(catalogue) + 1)
            if verbose:
                calibration_error = catalogue.meta["calibration_error"]
                print(f"{len(catalogue)} stars with {first_band} and {second_band}; calibration (systematic) uncertainty: {first_band} "
                      f"{calibration_error[first_band]:.3f}, {second_band} {calibration_error[second_band]:.3f}, {color_name} "
                      f"{calibration_error[color_name]:.3f} mag")
            return catalogue

        # Single band
        catalogues = []
        for band, table in tables_by_band.items():
            if band not in solution:
                raise ValueError(f"No solution for the band {band}.")
            coefficients, covariance = solution[band]["coefficients"], solution[band]["covariance"]
            colors = (np.zeros(len(table)) if color_index_values is None
                      else np.broadcast_to(np.asarray(color_index_values, dtype=float), (len(table),)))
            if color_index_values is None and abs(coefficients["c"]) > 0.02:
                warnings.warn(f"Band {band}: no colors given; the color term (c = {coefficients['c']:+.3f}) is neglected.")
            airmass = float(table.meta["airmass"])
            catalogue = Table()
            for name in ("id", "x", "y", "ra", "dec"):
                if name in table.colnames:
                    catalogue[name] = table[name]
            catalogue[band] = (np.asarray(table["instrumental_magnitude"], dtype=float) + coefficients["ZP"] - coefficients["k"]*airmass
                               - coefficients["c"]*colors)
            catalogue[band + "_error"] = table["instrumental_magnitude_error"]
            catalogue["flags"] = table["flags"]
            gradient = np.array([1.0, -airmass, -float(np.median(colors))])
            catalogue.meta["calibration_error"] = {band: float(np.hypot(np.sqrt(gradient @ covariance @ gradient),
                                                                        table.meta.get("aperture_correction_magnitude_error", 0.0)))}
            catalogue.meta["bands"] = (band,)
            catalogues.append(catalogue)
        return catalogues[0] if len(catalogues) == 1 else catalogues

    # ==================================================================================================================================
    # Surface photometry
    # ==================================================================================================================================

    def surface_brightness_profile(self, image, center="auto", ellipticity=0.0, position_angle=0.0, fit_isophotes=False,
                                   minimum_radius=None, maximum_radius=None, number_of_annuli=30, remove_stars="psf",
                                   add_back_resolved_stars=False, star_mask_threshold=1.0, isophote_smoothing=None, verbose=True):

        """
        Surface-brightness profile of an extended object (globular cluster, galaxy) in elliptical annuli.

        The resolved stars are removed before the profile is measured:
            - remove_stars="psf" (default): the PSF models of all stars fitted by psf_photometry() are subtracted, and the cores of the
              brightest stars, where the uncertainty of the subtracted model (relative PSF-model error, measured by psf_photometry(), x
              model) is above star_mask_threshold x the pixel noise, are masked (as are saturated stars);
            - remove_stars="mask": discs around the detected stars are masked (radius growing with the brightness of the star); simpler,
              but in crowded cores most pixels are masked and the stars below the detection threshold of find_stars() remain;
            - remove_stars=None: nothing is removed (integrated light of everything).
        The light of the stars below the detection limit remains in the image and is part of the measured surface brightness. Since the
        detection completeness drops towards the crowded centre of a cluster, the light of the unresolved stars alone is more concentrated
        than the cluster: to measure the structure of a cluster use add_back_resolved_stars=True, which adds the (noise-free) fitted fluxes
        of the subtracted stars back to their annuli and gives the total surface brightness. For a galaxy with foreground stars, keep it
        False. In each annulus the intensity of the image is the sigma-clipped mean of the remaining pixels. The sky is NOT subtracted: the profile fits
        (fit_king_profile(), fit_sersic_profile()) fit it together with the object, which is necessary when the object fills the field.

        Parameters
        ----------
        image: Image
            Image from load_images() (for remove_stars="psf", run psf_photometry() on it first).
        center: "auto" or (x, y)
            Centre in pixels (0-based). "auto" finds the maximum of the image (total light) smoothed with a Gaussian of 10 FWHM.
        ellipticity, position_angle: float
            Shape of the annuli: ellipticity = 1 - b/a, position angle in degrees counter-clockwise from the +x axis. With
            fit_isophotes=True they are measured (photutils.isophote) and the median values over the profile are used.
        fit_isophotes: bool
            If True, fit elliptical isophotes (Jedrzejewski 1987) to measure the centre, ellipticity and position angle; the isophote table
            is stored in profile.meta["isophotes"]. This works for smooth light distributions (galaxies); in partially resolved objects
            (globular clusters) the light is too grainy for a reliable shape, even with isophote_smoothing.
        isophote_smoothing: int or None
            Size (pixels) of a median filter applied to the image before the isophote fit (default: none).
        minimum_radius, maximum_radius: float or None
            Radii (pixels) of the first and last annulus (defaults: 1 FWHM and the largest radius within the image).
        number_of_annuli: int
            Number of logarithmically spaced annuli.
        remove_stars: "psf", "mask" or None
            How the resolved stars are removed (see above).
        add_back_resolved_stars: bool
            With remove_stars="psf": add the fitted fluxes of the subtracted stars back to their annuli (total surface brightness).
        star_mask_threshold: float
            Pixels where the uncertainty of the subtracted PSF models exceeds star_mask_threshold x the pixel noise are masked
            (remove_stars="psf").
        verbose: bool
            If True, print a summary.

        Returns
        -------
        profile: Table
            radius (semi-major axis, pixels and arcsec), inner and outer radius of the annulus, intensity and intensity_error (e-/s/arcsec^2,
            sky included), instrumental surface brightness (mag/arcsec^2), number of pixels and masked fraction. The image metadata,
            centre and shape are in profile.meta.
        """

        if image.background is None:
            self.find_stars(image, background_box_size=32, verbose=verbose)
        pixel_scale = aux.pixel_scale_of(image)
        if not np.isfinite(pixel_scale):
            raise ValueError(f"{image.name}: unknown pixel scale (no WCS and no PIXSCALE keyword).")
        fwhm = image.fwhm or 3.0
        data = np.array(image.data, dtype=float)
        mask = image.bad_pixels.copy()
        noise = np.sqrt(np.clip(image.variance, 1e-6, None))
        if remove_stars == "psf":
            if "star_model_image" not in image.extra:
                raise RuntimeError("Run psf_photometry() on this image first, or use remove_stars='mask'.")
            model_image = image.extra["star_model_image"]
            data = data - model_image
            # Where the uncertainty of the subtracted model (relative PSF-model error x model) exceeds the pixel noise, the residuals of
            # the stars dominate: those pixels (the cores of the bright stars) are masked
            photometry_table = image.extra.get("photometry")
            model_error_fraction = max(photometry_table.meta.get("aperture_correction_scatter", 0.0), 0.02) if photometry_table is not None \
                else 0.05
            mask |= model_error_fraction*model_image > star_mask_threshold*noise
            # Saturated stars are not well subtracted: their saturated pixels are masked, with a margin of 1 FWHM
            saturated_pixels = image.data > image.saturation_level
            if np.any(saturated_pixels):
                from scipy import ndimage
                mask |= ndimage.binary_dilation(saturated_pixels, iterations=max(int(round(fwhm)), 1))
        elif remove_stars == "mask":
            # Discs around the detected stars, larger for brighter stars (their wings): r = FWHM (1 + 0.6 log10(peak / background RMS))
            sources = image.extra["sources"]
            rms = float(np.median(image.background_rms))
            disc_radii = fwhm*(1 + 0.6*np.log10(np.clip(np.asarray(sources["peak"], dtype=float)/rms, 1, None)))
            source_positions = np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)]
            for disc_radius in np.unique(np.round(disc_radii*2)/2):
                selected = np.round(disc_radii*2)/2 == disc_radius
                mask |= aux.exclusion_mask(data.shape, source_positions[selected], disc_radius)
        elif remove_stars is not None:
            raise ValueError("remove_stars must be 'psf', 'mask' or None.")
        if isinstance(center, str) and center == "auto":
            # Centre of the total light (stars included), heavily smoothed: robust against the residuals of the subtracted stars
            center = aux_surface.light_center(image.data, image.bad_pixels, smoothing_sigma=max(10*fwhm, 20.0))
        number_of_rows, number_of_columns = data.shape
        maximum_radius = maximum_radius or min(center[0], center[1], number_of_columns - 1 - center[0], number_of_rows - 1 - center[1])
        minimum_radius = minimum_radius or fwhm
        isophote_table = None
        if fit_isophotes:
            isophotes = aux_surface.fit_isophotes(data, mask, center, initial_semi_major_axis=max(5*fwhm, 0.1*maximum_radius),
                                                  initial_ellipticity=max(ellipticity, 0.05), initial_position_angle_degrees=position_angle,
                                                  maximum_semi_major_axis=maximum_radius, smoothing_size=isophote_smoothing)
            if isophotes is None:
                warnings.warn(f"{image.name}: the isophote fit failed; circular/elliptical annuli with the given shape are used.")
            else:
                isophote_table = Table({"semi_major_axis": isophotes.sma, "intensity": isophotes.intens,
                                        "intensity_error": isophotes.int_err, "ellipticity": isophotes.eps,
                                        "ellipticity_error": isophotes.ellip_err, "position_angle": np.degrees(isophotes.pa) % 180,
                                        "position_angle_error": np.degrees(isophotes.pa_err), "x_center": isophotes.x0,
                                        "y_center": isophotes.y0, "stop_code": isophotes.stop_code})
                good = (isophote_table["stop_code"] <= 2) & (isophote_table["semi_major_axis"] > 2*fwhm)
                if np.count_nonzero(good) >= 3:
                    ellipticity = float(np.median(isophote_table["ellipticity"][good]))
                    position_angle = float(np.median(isophote_table["position_angle"][good]))
                    center = (float(np.median(isophote_table["x_center"][good])), float(np.median(isophote_table["y_center"][good])))
        edges = aux_surface.logarithmic_radius_edges(minimum_radius, maximum_radius, number_of_annuli)
        star_positions, star_fluxes = None, None
        if add_back_resolved_stars:
            if remove_stars != "psf":
                raise ValueError("add_back_resolved_stars=True needs remove_stars='psf'.")
            photometry_table = image.extra["photometry"]
            correction = photometry_table.meta.get("aperture_correction", 1.0)
            star_positions = np.c_[np.asarray(photometry_table["x"], dtype=float), np.asarray(photometry_table["y"], dtype=float)]
            # The model image was made with the PSF fluxes (before the PSF-to-total correction)
            star_fluxes = np.asarray(photometry_table["flux"], dtype=float)/correction
            usable_stars = np.all(np.isfinite(star_positions), axis=1) & np.isfinite(star_fluxes)
            star_positions, star_fluxes = star_positions[usable_stars], star_fluxes[usable_stars]
        values = aux_surface.annular_profile(data, image.variance, mask, center, edges, ellipticity, position_angle, fwhm=fwhm,
                                             star_positions=star_positions, star_fluxes=star_fluxes)
        area_per_pixel = pixel_scale**2
        profile = Table()
        profile["radius"] = values["radius"]
        profile["radius_arcsec"] = values["radius"]*pixel_scale
        profile["inner_radius"], profile["outer_radius"] = values["inner_radius"], values["outer_radius"]
        profile["intensity"] = values["intensity"]/image.exposure_time/area_per_pixel
        profile["intensity_error"] = values["intensity_error"]/image.exposure_time/area_per_pixel
        with np.errstate(divide="ignore", invalid="ignore"):
            profile["instrumental_surface_brightness"] = -2.5*np.log10(profile["intensity"])
        profile["resolved_star_intensity"] = values["resolved_star_intensity"]/image.exposure_time/area_per_pixel
        profile["number_of_pixels"] = values["number_of_pixels"]
        profile["masked_fraction"] = values["masked_fraction"]
        profile.meta.update({"image": image.name, "band": image.band, "airmass": image.airmass, "exposure_time": image.exposure_time,
                             "center": tuple(float(value) for value in center), "ellipticity": float(ellipticity),
                             "position_angle": float(position_angle), "pixel_scale": pixel_scale, "fwhm_arcsec": fwhm*pixel_scale,
                             "intensity_units": "e-/s/arcsec^2", "stars_removed": str(remove_stars),
                             "resolved_stars_added_back": bool(add_back_resolved_stars)})
        if isophote_table is not None:
            profile.meta["isophotes"] = isophote_table
        if verbose:
            print(f"{image.name}: profile in {len(profile)} annuli from {profile['radius_arcsec'][0]:.1f} to "
                  f"{profile['radius_arcsec'][-1]:.1f} arcsec; centre ({center[0]:.1f}, {center[1]:.1f}), ellipticity {ellipticity:.3f}, "
                  f"position angle {position_angle:.1f} deg")
        return profile

    def fit_king_profile(self, profile, fit_sky=True, fixed_parameters=None, minimum_radius=None, verbose=True):

        """
        Fit of a King (1962) profile plus a constant sky to a surface-brightness profile (globular and open clusters):

            I(r) = I_0 [ (1 + (r/r_c)^2)^(-1/2) - (1 + (r_t/r_c)^2)^(-1/2) ]^2 / [1 - (1 + (r_t/r_c)^2)^(-1/2)]^2 + sky.

        The tidal radius is only constrained if the profile reaches it; otherwise fix it with fixed_parameters={"tidal_radius": value}
        (arcsec) — the core radius and the central intensity are robust to that choice. When the cluster fills the field, the sky and
        the outer profile are degenerate: the fitted sky absorbs part of the outer cluster light.

        Parameters
        ----------
        profile: Table
            Output of surface_brightness_profile().
        fit_sky: bool
            Fit a constant sky (True) or assume the profile is sky-subtracted (False).
        fixed_parameters: dict or None
            Fixed values for "central_intensity" (e-/s/arcsec^2), "core_radius", "tidal_radius" (arcsec) or "sky".
        minimum_radius: float or None
            Inner radius (arcsec) of the fitted range (default: 1 FWHM, where the PSF does not matter).
        verbose: bool
            If True, print the result.

        Returns
        -------
        fit: dict
            "values", "errors" (central_intensity, core_radius, tidal_radius, sky), "reduced_chi2", "model" (function of the radius in
            arcsec) and the instrumental central surface brightness ("central_instrumental_surface_brightness").
        """

        radius, intensity, error = self._profile_arrays(profile, minimum_radius)
        positive = intensity - np.min(intensity)
        initial_core = float(np.interp(0.5*positive[0], positive[::-1], radius[::-1])) if positive[0] > 0 else radius[len(radius)//3]
        fit = aux_surface.fit_profile(radius, intensity, error, aux_surface.king_profile,
                                      ("central_intensity", "core_radius", "tidal_radius"),
                                      (max(positive[0], 1e-6), max(initial_core, radius[0]), 30*max(initial_core, radius[0])),
                                      (0, 0.1*radius[0], radius[0]), (np.inf, 10*radius[-1], 1e3*radius[-1]),
                                      fixed_values=fixed_parameters, fit_sky=fit_sky, initial_sky=float(np.min(intensity)))
        fit["central_instrumental_surface_brightness"] = float(-2.5*np.log10(fit["values"]["central_intensity"]))
        fit["profile_type"] = "king"
        fit["band"] = profile.meta.get("band")
        if verbose:
            values, errors = fit["values"], fit["errors"]
            print(f"King fit ({profile.meta.get('image')}): I_0 = {values['central_intensity']:.4g} +/- {errors['central_intensity']:.2g} "
                  f"e-/s/arcsec^2 (instrumental mu_0 = {fit['central_instrumental_surface_brightness']:.3f}), r_c = "
                  f"{values['core_radius']:.2f} +/- {errors['core_radius']:.2f} arcsec, r_t = {values['tidal_radius']:.1f} +/- "
                  f"{errors['tidal_radius']:.1f} arcsec, sky = {values['sky']:.4g} +/- {errors['sky']:.2g}; reduced chi2 = "
                  f"{fit['reduced_chi2']:.2f}")
            if "tidal_radius" in fit["free_parameters"] and values["tidal_radius"] > 2*profile["radius_arcsec"][-1]:
                print("   The tidal radius is far beyond the profile: it is not constrained by the data (fix it if known).")
        return fit

    def fit_sersic_profile(self, profile, fit_sky=True, fixed_parameters=None, minimum_radius=None, verbose=True):

        """
        Fit of a Sersic profile plus a constant sky to a surface-brightness profile (galaxies, bulges):

            I(r) = I_e exp(-b_n [(r/r_e)^(1/n) - 1]) + sky,   b_n such that r_e contains half of the light.

        The PSF is not deconvolved: radii smaller than ~3 FWHM are biased (excluded with minimum_radius, default 2 FWHM).

        Parameters
        ----------
        profile: Table
            Output of surface_brightness_profile().
        fit_sky: bool
            Fit a constant sky (True) or assume the profile is sky-subtracted (False).
        fixed_parameters: dict or None
            Fixed values for "effective_intensity", "effective_radius" (arcsec), "sersic_index" or "sky".
        minimum_radius: float or None
            Inner radius (arcsec) of the fitted range.
        verbose: bool
            If True, print the result.

        Returns
        -------
        fit: dict
            "values", "errors" (effective_intensity, effective_radius, sersic_index, sky), "reduced_chi2", "model" (function of the radius
            in arcsec) and the instrumental effective surface brightness ("effective_instrumental_surface_brightness").
        """

        minimum_radius = minimum_radius if minimum_radius is not None else 2*profile.meta.get("fwhm_arcsec", 0.0)
        radius, intensity, error = self._profile_arrays(profile, minimum_radius)
        positive = intensity - np.min(intensity)
        cumulative = np.cumsum(positive*2*np.pi*radius*np.gradient(radius))
        initial_effective_radius = float(np.interp(0.5*cumulative[-1], cumulative, radius))
        initial_effective_intensity = float(np.interp(initial_effective_radius, radius, positive))
        fit = aux_surface.fit_profile(radius, intensity, error, aux_surface.sersic_profile,
                                      ("effective_intensity", "effective_radius", "sersic_index"),
                                      (max(initial_effective_intensity, 1e-6), initial_effective_radius, 2.0),
                                      (0, 0.1*radius[0], 0.3), (np.inf, 20*radius[-1], 10.0),
                                      fixed_values=fixed_parameters, fit_sky=fit_sky, initial_sky=float(np.min(intensity)))
        fit["effective_instrumental_surface_brightness"] = float(-2.5*np.log10(fit["values"]["effective_intensity"]))
        fit["profile_type"] = "sersic"
        fit["band"] = profile.meta.get("band")
        if verbose:
            values, errors = fit["values"], fit["errors"]
            print(f"Sersic fit ({profile.meta.get('image')}): I_e = {values['effective_intensity']:.4g} +/- "
                  f"{errors['effective_intensity']:.2g} e-/s/arcsec^2, r_e = {values['effective_radius']:.2f} +/- "
                  f"{errors['effective_radius']:.2f} arcsec, n = {values['sersic_index']:.2f} +/- {errors['sersic_index']:.2f}, sky = "
                  f"{values['sky']:.4g}; reduced chi2 = {fit['reduced_chi2']:.2f}")
        return fit

    @staticmethod
    def _profile_arrays(profile, minimum_radius):
        """Radius (arcsec), intensity and error arrays of a profile, from minimum_radius (default: 1 FWHM) on."""
        radius = np.asarray(profile["radius_arcsec"], dtype=float)
        intensity = np.asarray(profile["intensity"], dtype=float)
        error = np.asarray(profile["intensity_error"], dtype=float)
        minimum_radius = profile.meta.get("fwhm_arcsec", 0.0) if minimum_radius is None else minimum_radius
        usable = (radius >= minimum_radius) & np.isfinite(intensity) & np.isfinite(error) & (error > 0)
        if np.count_nonzero(usable) < 5:
            raise ValueError("Fewer than 5 annuli to fit.")
        return radius[usable], intensity[usable], error[usable]

    def calibrate_surface_brightness(self, profiles, profile_fits=None, solution=None, color_index_value=None, verbose=True):

        """
        Calibrated surface-brightness profiles (mag/arcsec^2) and color profile.

        The sky fitted by fit_king_profile() / fit_sersic_profile() is subtracted, and the intensities are calibrated with the standard
        star solution exactly as stellar magnitudes, with m_inst = -2.5 log10(intensity [e-/s/arcsec^2]). With two profiles (the bands of
        solution["color_index"]), the color of each annulus is solved together with the surface brightness; the profiles are interpolated
        to the radii of the first one. The central (King) or effective (Sersic) surface brightness of the fits is also calibrated.

        Parameters
        ----------
        profiles: Table or list of Table
            Outputs of surface_brightness_profile() (one per band, same centre and shape).
        profile_fits: dict, list of dict or None
            Fits of each profile (same order), whose sky is subtracted. Without fits the profiles must be sky-subtracted already.
        solution: dict
            Output of fit_standard_star_solution() (default: the last one computed).
        color_index_value: float or None
            For a single band: color index of the object, for the color term (default 0).
        verbose: bool
            If True, print the calibrated central/effective surface brightness.

        Returns
        -------
        calibrated: Table
            radius_arcsec, surface brightness per band (<band>, mag/arcsec^2) with errors, and the color (two bands). The calibrated
            central or effective surface brightness of the fits is in calibrated.meta["fitted_surface_brightness"].
        """

        solution = solution or self.standard_star_solution
        if solution is None:
            raise ValueError("No photometric solution: run fit_standard_star_solution() first.")
        profiles = [profiles] if isinstance(profiles, Table) else list(profiles)
        if profile_fits is None:
            profile_fits = [None]*len(profiles)
        elif isinstance(profile_fits, dict):
            profile_fits = [profile_fits]
        bands = [profile.meta["band"] for profile in profiles]
        radius = np.asarray(profiles[0]["radius_arcsec"], dtype=float)
        intensities, errors, fitted_reference = {}, {}, {}
        for profile, fit, band in zip(profiles, profile_fits, bands):
            sky = fit["values"]["sky"] if fit is not None else 0.0
            sky_error = fit["errors"]["sky"] if fit is not None else 0.0
            intensity = np.interp(radius, profile["radius_arcsec"], profile["intensity"]) - sky
            error = np.hypot(np.interp(radius, profile["radius_arcsec"], profile["intensity_error"]), sky_error)
            intensities[band], errors[band] = intensity, error
            if fit is not None:
                key = "central_intensity" if fit["profile_type"] == "king" else "effective_intensity"
                fitted_reference[band] = (fit["values"][key], fit["errors"][key], key)
        calibrated = Table()
        calibrated["radius_arcsec"] = radius
        with np.errstate(divide="ignore", invalid="ignore"):
            instrumental = {band: -2.5*np.log10(intensities[band]) for band in bands}
            instrumental_error = {band: MAGNITUDE_ERROR_FACTOR*errors[band]/intensities[band] for band in bands}
        first_band, second_band = solution["color_index"]
        fitted_surface_brightness = {}
        if len(bands) == 2 and set(bands) == {first_band, second_band}:
            airmass = {profile.meta["band"]: float(profile.meta["airmass"]) for profile in profiles}
            magnitude_1, magnitude_2, color, error_1, error_2, color_error, _ = aux.solve_two_band_magnitudes(
                instrumental[first_band], instrumental[second_band], airmass[first_band], airmass[second_band], solution[first_band],
                solution[second_band], instrumental_error[first_band], instrumental_error[second_band])
            calibrated[first_band], calibrated[first_band + "_error"] = magnitude_1, error_1
            calibrated[second_band], calibrated[second_band + "_error"] = magnitude_2, error_2
            calibrated[f"{first_band}-{second_band}"], calibrated[f"{first_band}-{second_band}_error"] = color, color_error
            if len(fitted_reference) == 2 and fitted_reference[first_band][2] == fitted_reference[second_band][2]:
                reference_1 = -2.5*np.log10(fitted_reference[first_band][0])
                reference_2 = -2.5*np.log10(fitted_reference[second_band][0])
                fitted_1, fitted_2, fitted_color, fitted_error_1, fitted_error_2, _, _ = aux.solve_two_band_magnitudes(
                    np.array([reference_1]), np.array([reference_2]), airmass[first_band], airmass[second_band], solution[first_band],
                    solution[second_band], np.array([MAGNITUDE_ERROR_FACTOR*fitted_reference[first_band][1]/fitted_reference[first_band][0]]),
                    np.array([MAGNITUDE_ERROR_FACTOR*fitted_reference[second_band][1]/fitted_reference[second_band][0]]))
                kind = "central" if fitted_reference[first_band][2] == "central_intensity" else "effective"
                fitted_surface_brightness = {"kind": kind, first_band: (float(fitted_1[0]), float(fitted_error_1[0])),
                                             second_band: (float(fitted_2[0]), float(fitted_error_2[0])),
                                             f"{first_band}-{second_band}": float(fitted_color[0])}
        else:
            for profile, band in zip(profiles, bands):
                coefficients = solution[band]["coefficients"]
                color_value = color_index_value or 0.0
                airmass = float(profile.meta["airmass"])
                calibrated[band] = instrumental[band] + coefficients["ZP"] - coefficients["k"]*airmass - coefficients["c"]*color_value
                calibrated[band + "_error"] = instrumental_error[band]
                if band in fitted_reference:
                    value, value_error, key = fitted_reference[band]
                    fitted_surface_brightness[band] = (float(-2.5*np.log10(value) + coefficients["ZP"] - coefficients["k"]*airmass
                                                             - coefficients["c"]*color_value),
                                                       float(MAGNITUDE_ERROR_FACTOR*value_error/value))
                    fitted_surface_brightness["kind"] = "central" if key == "central_intensity" else "effective"
        calibrated.meta["fitted_surface_brightness"] = fitted_surface_brightness
        calibrated.meta["units"] = "mag/arcsec^2"
        if verbose and fitted_surface_brightness:
            kind = fitted_surface_brightness["kind"]
            text = ", ".join(f"{band} = {value[0]:.3f} +/- {value[1]:.3f}" for band, value in fitted_surface_brightness.items()
                             if band in bands)
            color_text = (f", {first_band}-{second_band} = {fitted_surface_brightness[f'{first_band}-{second_band}']:.3f}"
                          if f"{first_band}-{second_band}" in fitted_surface_brightness else "")
            print(f"Calibrated {kind} surface brightness: {text} mag/arcsec^2{color_text}")
        return calibrated

    def plot_surface_brightness_profile(self, profile, profile_fit=None, calibrated_profile=None, band=None, title=None):

        """
        Plot a surface-brightness profile: instrumental intensity (with the fit and the fitted sky) and, if given, the calibrated
        surface brightness (mag/arcsec^2) and the color profile.
        """

        number_of_panels = 1 + (calibrated_profile is not None)
        figure, axes = plt.subplots(1, number_of_panels, figsize=(6.5*number_of_panels, 5), squeeze=False)
        axis = axes[0, 0]
        radius = np.asarray(profile["radius_arcsec"])
        axis.errorbar(radius, profile["intensity"], profile["intensity_error"], fmt="o", ms=4, color="k", label="data")
        if profile_fit is not None:
            fine_radius = np.geomspace(radius[0], radius[-1], 300)
            axis.plot(fine_radius, profile_fit["model"](fine_radius), color="C3", label=f"{profile_fit['profile_type'].capitalize()} + sky")
            axis.axhline(profile_fit["values"]["sky"], color="C0", ls="--", lw=1, label="fitted sky")
            if profile_fit["profile_type"] == "king":
                axis.axvline(profile_fit["values"]["core_radius"], color="C3", ls=":", lw=1, label="core radius")
            else:
                axis.axvline(profile_fit["values"]["effective_radius"], color="C3", ls=":", lw=1, label="effective radius")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Semi-major axis (arcsec)")
        axis.set_ylabel(r"Intensity (e$^-$ s$^{-1}$ arcsec$^{-2}$)")
        axis.legend(fontsize=8)
        axis.set_title(title or profile.meta.get("image", ""))
        if calibrated_profile is not None:
            axis = axes[0, 1]
            bands = [name for name in calibrated_profile.colnames if name not in ("radius_arcsec",) and not name.endswith("_error")
                     and "-" not in name]
            for index, band_name in enumerate(bands if band is None else [band]):
                axis.errorbar(calibrated_profile["radius_arcsec"], calibrated_profile[band_name], calibrated_profile[band_name + "_error"],
                              fmt="o-", ms=3, color=f"C{index}", label=band_name)
            axis.invert_yaxis()
            axis.set_xscale("log")
            axis.set_xlabel("Semi-major axis (arcsec)")
            axis.set_ylabel(r"Surface brightness (mag arcsec$^{-2}$)")
            color_names = [name for name in calibrated_profile.colnames if "-" in name and not name.endswith("_error")]
            if color_names:
                color_axis = axis.twinx()
                color_axis.plot(calibrated_profile["radius_arcsec"], calibrated_profile[color_names[0]], "s--", ms=3, color="0.5")
                color_axis.set_ylabel(color_names[0], color="0.5")
            axis.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        plt.show()

    def surface_photometry(self, images, solution=None, profile_model="king", fixed_parameters=None, remove_stars="psf",
                           add_back_resolved_stars=False, center="auto", ellipticity=0.0, position_angle=0.0, fit_isophotes=False,
                           minimum_radius=None, maximum_radius=None, number_of_annuli=30, save_results=True, output_directory=".",
                           file_prefix=None, plot=True, plot_annuli=False, image_scale="log", verbose=True):

        """
        Surface photometry of an extended object in one or more bands, in one call: surface-brightness profile of each image, fit of a
        King or Sersic profile (plus sky), calibration in mag/arcsec^2 with the standard-star solution (and the color profile, with two
        bands), files with the results, and plots.

        It runs, for each image, surface_brightness_profile() and fit_king_profile() / fit_sersic_profile(), and then
        calibrate_surface_brightness() (see these methods for the details).

        Parameters
        ----------
        images: Image or list of Image
            Images of the object (one per band). For remove_stars="psf", run psf_photometry() on them first.
        solution: dict or None
            Standard-star solution (fit_standard_star_solution()); default: the last one computed. Without a solution the profiles are
            not calibrated.
        profile_model: "king", "sersic" or None
            Profile fitted to each band: King (1962) for clusters, Sersic for galaxies, or None (no fit).
        fixed_parameters: dict or None
            Parameters fixed in the fit, e.g. {"tidal_radius": 720.0} (arcsec) for King, or {"sersic_index": 4.0}. Use it for parameters
            the data cannot constrain (e.g. a tidal radius far beyond the field).
        remove_stars: "psf", "mask" or None
            How the resolved stars are removed before measuring the light in annuli (see surface_brightness_profile()).
        add_back_resolved_stars: bool
            If True, the fitted fluxes of the subtracted stars are added back to their annuli: the profile is the total surface brightness
            (resolved + unresolved stars), the right choice for the structure (King profile) of a star cluster. If False (default), only
            the unresolved light is measured (e.g. a galaxy with foreground stars removed).
        center, ellipticity, position_angle, fit_isophotes, minimum_radius, maximum_radius, number_of_annuli:
            Geometry of the annuli (see surface_brightness_profile()). With fit_isophotes=True the ellipticity and position angle are
            measured from elliptical isophotes (galaxies).
        save_results: bool
            If True (default), save the results as ECSV tables (plain text with the metadata in the header; read them with
            astropy.table.Table.read(file_name)) in output_directory:
            - <prefix>_surface_profile_<band>.ecsv: the profile of each band (one row per annulus), with the fit in the header;
            - <prefix>_surface_profile_fits.ecsv: the fitted parameters and errors (one row per band);
            - <prefix>_surface_profile_calibrated.ecsv: surface brightness (mag/arcsec^2) and color profile, if calibrated.
        output_directory: str or Path
            Folder of the files (created if needed); default: the current folder.
        file_prefix: str or None
            Beginning of the file names; default: the object name of the first image (header OBJECT), or "surface_photometry".
        plot: bool
            If True (default), plot the profiles of all bands (intensity with the fits and the fitted sky) and, if calibrated, the surface
            brightness and the color profile.
        plot_annuli: bool
            If True, show each image with the annuli (circular or elliptical) in which the surface brightness is measured, the centre and
            the core (King) or effective (Sersic) radius.
        image_scale: "log" or "linear"
            Intensity scale of the images in the annuli plots.
        verbose: bool
            If True, print the results.

        Returns
        -------
        result: dict
            "profiles": {band: profile table}, with radius_arcsec, inner_radius, outer_radius, intensity, intensity_error (e-/s/arcsec^2,
            sky included), ...;
            "fits": {band: fit} (values, errors, model(radius), ...);
            "calibrated_profile": table with the surface brightness of each band (mag/arcsec^2), the color and their errors (None if
            there is no solution);
            "fitted_surface_brightness": calibrated central (King) or effective (Sersic) surface brightness of each band;
            "files": list of the saved files.
        """

        images = [images] if isinstance(images, aux.Image) else list(images)
        if profile_model not in ("king", "sersic", None):
            raise ValueError("profile_model must be 'king', 'sersic' or None.")
        solution = solution or self.standard_star_solution
        profiles, fits, images_by_band = {}, {}, {}
        for image in images:
            band = image.band or image.name
            profile = self.surface_brightness_profile(image, center=center, ellipticity=ellipticity, position_angle=position_angle,
                                                      fit_isophotes=fit_isophotes, minimum_radius=minimum_radius,
                                                      maximum_radius=maximum_radius, number_of_annuli=number_of_annuli,
                                                      remove_stars=remove_stars, add_back_resolved_stars=add_back_resolved_stars,
                                                      verbose=verbose)
            profiles[band] = profile
            images_by_band[band] = image
            if profile_model == "king":
                fits[band] = self.fit_king_profile(profile, fixed_parameters=fixed_parameters, verbose=verbose)
            elif profile_model == "sersic":
                fits[band] = self.fit_sersic_profile(profile, fixed_parameters=fixed_parameters, verbose=verbose)
        calibrated_profile = None
        if solution is not None and all(band in solution for band in profiles):
            calibrated_profile = self.calibrate_surface_brightness(list(profiles.values()),
                                                                   [fits[band] for band in profiles] if fits else None, solution,
                                                                   verbose=verbose)
        elif verbose:
            print("No standard-star solution for these bands: the profiles are not calibrated.")
        result = {"profiles": profiles, "fits": fits, "calibrated_profile": calibrated_profile,
                  "fitted_surface_brightness": calibrated_profile.meta.get("fitted_surface_brightness", {}) if calibrated_profile is not None
                  else {}, "files": []}
        if save_results:
            prefix = file_prefix or images[0].object_name or "surface_photometry"
            result["files"] = self._save_surface_photometry(result, Path(output_directory), str(prefix).replace(" ", "_"))
            if verbose:
                print("Saved: " + ", ".join(str(file_name) for file_name in result["files"]))
        if plot:
            self._plot_surface_photometry(result)
        if plot_annuli:
            for band, profile in profiles.items():
                self._plot_annuli(images_by_band[band], profile, fits.get(band), image_scale)
        return result

    def _save_surface_photometry(self, result, output_directory, prefix):
        """Writes the profiles, the fitted parameters and the calibrated profile of surface_photometry() as ECSV tables."""
        output_directory.mkdir(parents=True, exist_ok=True)
        files = []
        fit_rows = []
        for band, profile in result["profiles"].items():
            table = profile.copy()
            fit = result["fits"].get(band)
            if fit is not None:
                table.meta["fit"] = {"profile_type": fit["profile_type"], "reduced_chi2": float(fit["reduced_chi2"]),
                                     "values": {name: float(value) for name, value in fit["values"].items()},
                                     "errors": {name: float(value) for name, value in fit["errors"].items()},
                                     "free_parameters": list(fit["free_parameters"])}
                row = {"band": band, "profile_type": fit["profile_type"]}
                for name, value in fit["values"].items():
                    row[name] = float(value)
                    row[name + "_error"] = float(fit["errors"][name])
                    row[name + "_fixed"] = name not in fit["free_parameters"]
                row["reduced_chi2"] = float(fit["reduced_chi2"])
                calibrated = result["fitted_surface_brightness"].get(band)
                if calibrated is not None:
                    kind = result["fitted_surface_brightness"]["kind"]
                    row[f"{kind}_surface_brightness"], row[f"{kind}_surface_brightness_error"] = calibrated
                fit_rows.append(row)
            files.append(self.save_table(table, output_directory/f"{prefix}_surface_profile_{band}.ecsv"))
        if fit_rows:
            fit_table = Table(rows=fit_rows)
            for name in fit_table.colnames:
                if name.endswith(("radius", "radius_error")):
                    fit_table[name].unit = "arcsec"
                elif name.endswith(("intensity", "intensity_error", "sky", "sky_error")):
                    fit_table[name].unit = "electron / (s arcsec2)"
            files.append(self.save_table(fit_table, output_directory/f"{prefix}_surface_profile_fits.ecsv"))
        if result["calibrated_profile"] is not None:
            files.append(self.save_table(result["calibrated_profile"], output_directory/f"{prefix}_surface_profile_calibrated.ecsv"))
        return files

    @staticmethod
    def _plot_surface_photometry(result):
        """Intensity profiles of all bands (with fits and fitted sky) and, if calibrated, surface brightness and color profiles."""
        profiles, fits, calibrated = result["profiles"], result["fits"], result["calibrated_profile"]
        band_colors = {"U": "#8E44AD", "B": "#2E86C1", "V": "#27AE60", "R": "#E67E22", "I": "#C0392B"}
        number_of_panels = 2 if calibrated is not None else 1
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            figure, axes = plt.subplots(1, number_of_panels, figsize=(7.2*number_of_panels, 6), squeeze=False)
            axis = axes[0, 0]
            for index, (band, profile) in enumerate(profiles.items()):
                color = band_colors.get(band, f"C{index}")
                radius = np.asarray(profile["radius_arcsec"])
                axis.errorbar(radius, profile["intensity"], profile["intensity_error"], fmt="o", ms=4, color=color, capsize=2,
                              label=f"{band}")
                fit = fits.get(band)
                if fit is not None:
                    fine_radius = np.geomspace(radius[0], radius[-1], 300)
                    axis.plot(fine_radius, fit["model"](fine_radius), color=color, lw=1.6)
                    axis.axhline(fit["values"]["sky"], color=color, ls="--", lw=0.9, alpha=0.8)
                    characteristic = fit["values"].get("core_radius", fit["values"].get("effective_radius"))
                    axis.axvline(characteristic, color=color, ls=":", lw=1.0)
            if fits:
                model_name = next(iter(fits.values()))["profile_type"].capitalize()
                radius_name = "core radius" if model_name == "King" else "effective radius"
                axis.plot([], [], color="0.3", lw=1.6, label=f"{model_name} + sky fit")
                axis.plot([], [], color="0.3", ls="--", lw=0.9, label="fitted sky")
                axis.plot([], [], color="0.3", ls=":", lw=1.0, label=radius_name)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Semi-major axis (arcsec)")
            axis.set_ylabel(r"Intensity (e$^-$ s$^{-1}$ arcsec$^{-2}$)")
            axis.set_title("Surface-brightness profiles (instrumental)", fontsize=12)
            if calibrated is not None:
                axis = axes[0, 1]
                bands = [band for band in profiles if band in calibrated.colnames]
                for index, band in enumerate(bands):
                    axis.errorbar(calibrated["radius_arcsec"], calibrated[band], calibrated[band + "_error"], fmt="o-", ms=4, lw=1,
                                  capsize=2, color=band_colors.get(band, f"C{index}"), label=band)
                axis.invert_yaxis()
                axis.set_xscale("log")
                axis.set_xlabel("Semi-major axis (arcsec)")
                axis.set_ylabel(r"Surface brightness (mag arcsec$^{-2}$)")
                axis.set_title("Calibrated surface brightness", fontsize=12)
                color_names = [name for name in calibrated.colnames if "-" in name and not name.endswith("_error")]
                if color_names:
                    color_axis = axis.twinx()
                    color_axis.errorbar(calibrated["radius_arcsec"], calibrated[color_names[0]], calibrated[color_names[0] + "_error"],
                                        fmt="s--", ms=3, lw=0.8, capsize=2, color="0.45", label=color_names[0])
                    color_axis.set_ylabel(f"${color_names[0]}$ (mag)", color="0.35")
                    color_axis.legend(loc="lower right", fontsize=9)
            for axis in axes[0]:
                axis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.5)
                axis.tick_params(which="both", direction="in", top=True)
                axis.legend(loc="lower left", fontsize=9, framealpha=0.85)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def _plot_annuli(image, profile, fit=None, image_scale="log"):
        """Image with the annuli (circular or elliptical) in which the surface brightness was measured, the centre and the core/effective radius."""
        from matplotlib.patches import Ellipse as EllipsePatch
        center = profile.meta["center"]
        ellipticity = profile.meta.get("ellipticity", 0.0)
        position_angle = profile.meta.get("position_angle", 0.0)
        pixel_scale = profile.meta["pixel_scale"]
        edges = np.unique(np.r_[np.asarray(profile["inner_radius"], dtype=float), np.asarray(profile["outer_radius"], dtype=float)])

        def draw_ellipse(axis, semi_major_axis, **style):
            axis.add_patch(EllipsePatch(center, 2*semi_major_axis, 2*semi_major_axis*(1 - ellipticity), angle=position_angle, fill=False,
                                        **style))

        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            figure, axis = plt.subplots(figsize=(9.5, 9))
            shown = axis.imshow(image.data, origin="lower", cmap="magma", norm=_image_normalization(image.data, image_scale, 99.5),
                                interpolation="nearest")
            color_bar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
            color_bar.set_label(r"e$^-$ (" + image_scale + " scale)")
            for edge in edges:
                draw_ellipse(axis, edge, color="#4FC3F7", lw=0.8, alpha=0.9)
            axis.plot([], [], color="#4FC3F7", lw=0.8, label=f"annuli edges ({len(profile)} annuli)")
            if fit is not None:
                key = "core_radius" if fit["profile_type"] == "king" else "effective_radius"
                draw_ellipse(axis, fit["values"][key]/pixel_scale, color="#2ECC71", lw=1.8, ls="--")
                axis.plot([], [], color="#2ECC71", lw=1.8, ls="--", label=f"{key.replace('_', ' ')} ({fit['values'][key]:.1f}\")")
            axis.plot(*center, "+", color="#2ECC71", ms=14, mew=1.6, label="centre")
            axis.set_xlim(-0.5, image.data.shape[1] - 0.5)
            axis.set_ylim(-0.5, image.data.shape[0] - 0.5)
            # The inner annuli are thin (logarithmic spacing): zoom on them in an inset
            zoom_radius = 1.15*edges[len(edges)//2]
            if zoom_radius < 0.25*min(image.data.shape):
                inset = axis.inset_axes([0.62, 0.02, 0.36, 0.36])
                rows = slice(max(int(center[1] - zoom_radius), 0), int(center[1] + zoom_radius) + 1)
                columns = slice(max(int(center[0] - zoom_radius), 0), int(center[0] + zoom_radius) + 1)
                inset.imshow(image.data, origin="lower", cmap="magma", interpolation="nearest",
                             norm=_image_normalization(image.data[rows, columns], image_scale, 99.9, log_strength=30))
                for edge in edges[edges <= zoom_radius*1.5]:
                    draw_ellipse(inset, edge, color="#4FC3F7", lw=0.7, alpha=0.9)
                if fit is not None:
                    draw_ellipse(inset, fit["values"][key]/pixel_scale, color="#2ECC71", lw=1.5, ls="--")
                inset.plot(*center, "+", color="#2ECC71", ms=10, mew=1.2)
                inset.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
                inset.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_edgecolor("white")
                axis.indicate_inset_zoom(inset, edgecolor="white", alpha=0.8)
            shape_text = "circular" if ellipticity == 0 else f"elliptical, e = {ellipticity:.2f}, PA = {position_angle:.0f} deg"
            axis.set_title(f"{image.name}: annuli of the surface-brightness profile ({shape_text})", fontsize=12)
            axis.set_xlabel("x (pixels)")
            axis.set_ylabel("y (pixels)")
            axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.4, color="white")
            axis.tick_params(direction="in", top=True, right=True, color="0.6")
            axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
            plt.tight_layout()
            plt.show()

    # ==================================================================================================================================
    # Differential photometry (time series)
    # ==================================================================================================================================

    def differential_photometry(self, images, target_position=None, target_coordinates=None, comparison_stars="auto",
                                aperture_radii=None, sky_annulus=(3.0, 5.0), minimum_comparison_brightness=0.1, minimum_transparency=0.7,
                                time_scale="auto", detection_threshold=10.0, plot=True, image_scale="log", verbose=True):

        """
        Differential light curve of a target (e.g. an exoplanet host star) relative to an ensemble of comparison stars of the same field.

        The frames are measured as they are (not aligned, so the noise of each pixel stays independent): the field drift is found by a vote
        on the pairwise differences of the brightest stars relative to a reference frame (the one with most detections), and the positions
        are refined with the centroids of the brightest stars. Every star of the reference frame is measured in several apertures with a
        local sky annulus. The comparison stars, the frames (clouds, cosmic rays) and the aperture radius are then chosen automatically,
        using the point-to-point scatter, which does not depend on the transit (see aux_differential_photometry.build_differential_light_curve).

        Defocused images (often used for bright transit hosts) are handled: the FWHM is measured in each frame, and the apertures and
        annulus scale with the median FWHM of the series.

        Parameters
        ----------
        images: list of Image
            The time series (output of load_images() of frames reduced with cleaning; do not combine or align them).
        target_position: (x, y) or None
            Pixel position (0-based) of the target in the reference frame (the frame with most detected stars; it is the first frame if
            all have similar numbers). The closest detected star is used.
        target_coordinates: SkyCoord or (ra, dec) in degrees, or None
            Alternatively, the coordinates of the target (needs a WCS in the reference frame).
        comparison_stars: "auto" or list
            How the comparison stars are chosen:
              - "auto" (default): all stars of the reference frame measured in at least 90% of the frames, never close to saturation, and
                brighter than minimum_comparison_brightness x the target are candidates; each candidate is divided by the sum of the
                others, and the noisiest one (point-to-point scatter more than twice the median of the candidates, relative to its photon
                noise: variable stars, bad pixels, blends) is removed, repeatedly;
              - a list of stars chosen by the user, given as pixel positions (x, y) in the reference frame, as SkyCoord objects (needs a
                WCS), or as the numbers of the stars shown by plot_differential_photometry() in a previous run (they do not change between
                runs on the same frames).
                Each position is matched to the closest detected star. All the given stars are used (none is rejected), but their noise
                ratios are printed so that a bad choice can be spotted.
            In both cases the ensemble is the weighted sum of the comparisons (weights = inverse relative variance).
        aperture_radii: array or None
            Aperture radii to try, in units of the median FWHM (default 0.6 to 2.5 FWHM).
        sky_annulus: tuple
            Inner and outer radii of the sky annulus, in units of the median FWHM.
        minimum_comparison_brightness: float
            Comparison stars fainter than this fraction of the target are not used.
        minimum_transparency: float
            Frames where the summed flux of the comparisons is below this fraction of its median (clouds) are not used.
        time_scale: string
            "auto" (BJD_TDB if the target coordinates and the observatory location are known, otherwise MJD_UTC), "bjd_tdb", "mjd_utc"
            or "jd_utc". The times are mid-exposure.
        detection_threshold: float
            Detection threshold (sigma) used to find the stars of each frame for the registration.
        plot: bool
            If True (default), show the reference frame with the target (red) and the comparison stars (green); see
            plot_differential_photometry().
        image_scale: "log" or "linear"
            Intensity scale of the plot.
        verbose: bool
            If True, print a summary.

        Returns
        -------
        light_curve: Table
            One row per frame: time, relative_flux, relative_flux_error, airmass, fwhm, sky, x, y (target), transparency, used (False for
            rejected frames), image. The time scale, chosen aperture, comparison stars and the full measurement arrays are in
            light_curve.meta.
        """

        images = list(images)
        if len(images) < 10:
            raise ValueError("Differential photometry needs a time series (at least 10 frames).")
        # Stars and FWHM in each frame
        detections = []
        for image in images:
            sources = self.find_stars(image, detection_threshold=detection_threshold, verbose=False)
            order = np.argsort(-np.asarray(sources["flux_estimate"], dtype=float))
            detections.append(np.c_[np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float)][order])
        fwhm_per_frame = np.array([image.fwhm for image in images])
        median_fwhm = float(np.nanmedian(fwhm_per_frame))
        reference_index = int(np.argmax([len(positions) for positions in detections]))
        reference_image, reference_positions = images[reference_index], detections[reference_index]
        # Translation of each frame relative to the reference
        shifts = np.array([aux.estimate_translation(reference_positions, positions, number_of_bright_sources=40,
                                                    bin_size=max(1.0, median_fwhm/3)) for positions in detections])
        registered = shifts[:, 2] >= 3
        # Target
        if target_coordinates is not None:
            if reference_image.wcs is None:
                raise ValueError("target_coordinates needs a WCS in the reference frame; give target_position instead.")
            if not isinstance(target_coordinates, SkyCoord):
                target_coordinates = SkyCoord(*target_coordinates, unit="deg")
            target_position = np.array(reference_image.wcs.celestial.world_to_pixel(target_coordinates), dtype=float)
        if target_position is None:
            raise ValueError("Give the target: target_position=(x, y) in pixels of the reference frame, or target_coordinates=(ra, dec). "
                             "There is no automatic choice: the brightest star of the field is not necessarily the target.")
        distances = np.hypot(*(reference_positions - np.asarray(target_position, dtype=float)).T)
        target_index_in_reference = int(np.argmin(distances))
        if distances[target_index_in_reference] > 2*median_fwhm:
            raise ValueError(f"No star within 2 FWHM of the target position {tuple(target_position)} in the reference frame "
                             f"{reference_image.name} (closest at {distances[target_index_in_reference]:.1f} px).")
        star_positions = np.vstack([reference_positions[target_index_in_reference],
                                    np.delete(reference_positions, target_index_in_reference, axis=0)])   # star 0 = target
        if target_coordinates is None and reference_image.wcs is not None:
            target_coordinates = reference_image.wcs.celestial.pixel_to_world(*star_positions[0])
        # Measurements
        aperture_radii = np.arange(0.6, 2.55, 0.1) if aperture_radii is None else np.asarray(aperture_radii, dtype=float)
        radii_in_pixels = aperture_radii*median_fwhm
        annulus_in_pixels = (sky_annulus[0]*median_fwhm, sky_annulus[1]*median_fwhm)
        number_of_frames, number_of_stars = len(images), len(star_positions)
        fluxes = np.full((number_of_frames, number_of_stars, len(radii_in_pixels)), np.nan)
        errors = np.full_like(fluxes, np.nan)
        positions = np.full((number_of_frames, number_of_stars, 2), np.nan)
        sky = np.full((number_of_frames, number_of_stars), np.nan)
        peaks = np.full((number_of_frames, number_of_stars), np.nan)
        centroid_half_size = int(np.ceil(max(median_fwhm, 3.0)))
        for frame_index in np.flatnonzero(registered):
            predicted = star_positions - shifts[frame_index, :2]
            (positions[frame_index], fluxes[frame_index], errors[frame_index], sky[frame_index],
             peaks[frame_index]) = aux_differential.measure_frame(images[frame_index], predicted, radii_in_pixels, annulus_in_pixels,
                                                                  centroid_half_size)
        saturation_level = float(np.min([image.saturation_level for image in images]))
        aux_differential.warn_if_saturated(peaks, saturation_level)
        comparison_indices = None
        if not (isinstance(comparison_stars, str) and comparison_stars == "auto"):
            comparison_indices = self._comparison_indices(comparison_stars, star_positions, reference_image, median_fwhm)
        selection = aux_differential.build_differential_light_curve(
            fluxes, errors, peaks, saturation_level, comparison_indices=comparison_indices,
            minimum_relative_brightness=minimum_comparison_brightness, minimum_transparency=minimum_transparency,
            reject_noisy_comparisons=comparison_indices is None)
        times, scale_name = aux_differential.mid_exposure_times(images, target_coordinates, time_scale)
        radius_index = selection["radius_index"]
        light_curve = Table()
        light_curve["time"] = times
        light_curve["relative_flux"] = selection["relative_flux"]
        light_curve["relative_flux_error"] = selection["relative_flux_error"]
        light_curve["airmass"] = [image.airmass if image.airmass is not None else np.nan for image in images]
        light_curve["fwhm"] = fwhm_per_frame
        light_curve["sky"] = sky[:, 0]
        light_curve["x"], light_curve["y"] = positions[:, 0, 0], positions[:, 0, 1]
        light_curve["transparency"] = selection["transparency"]
        light_curve["used"] = selection["frames"]
        light_curve["image"] = [image.name for image in images]
        light_curve.meta.update({"time_scale": scale_name, "aperture_radius_fwhm": float(aperture_radii[radius_index]),
                                 "aperture_radius_pixels": float(radii_in_pixels[radius_index]), "sky_annulus_pixels": annulus_in_pixels,
                                 "median_fwhm": median_fwhm, "comparison_stars": [int(star) for star in selection["comparisons"]],
                                 "comparison_weights": [float(weight) for weight in selection["weights"]],
                                 "comparison_candidates": [int(star) for star in selection["candidates"]],
                                 "point_to_point_scatter": float(selection["point_to_point_scatter"]),
                                 "comparison_systematics": float(selection["comparison_systematics"]),
                                 "reference_frame": reference_image.name, "star_positions": star_positions,
                                 "reference_image": reference_image, "comparison_choice": "auto" if comparison_indices is None else "manual",
                                 "comparison_noise_ratio": selection.get("comparison_noise_ratio", {}),
                                 "exposure_time": float(np.median([image.exposure_time for image in images])),
                                 "band": reference_image.band, "object": reference_image.object_name,
                                 "number_of_outliers": int(np.count_nonzero(selection["outliers"])),
                                 "measurements": {"flux": fluxes, "flux_error": errors, "positions": positions, "sky": sky, "peaks": peaks,
                                                  "aperture_radii_pixels": radii_in_pixels}})
        if verbose:
            used = selection["frames"]
            photon_noise = np.nanmedian(selection["relative_flux_error"][used])
            print(f"Differential photometry of {number_of_frames} frames (reference {reference_image.name}, median FWHM "
                  f"{median_fwhm:.1f} px): target at ({star_positions[0, 0]:.1f}, {star_positions[0, 1]:.1f}); "
                  f"{len(selection['comparisons'])} comparison stars (of {len(selection['candidates'])} candidates); aperture "
                  f"{aperture_radii[radius_index]:.1f} FWHM = {radii_in_pixels[radius_index]:.1f} px; {np.count_nonzero(used)} frames used; "
                  f"scatter {1e3*selection['point_to_point_scatter']:.2f} ppt per frame (expected from photon noise "
                  f"{1e3*photon_noise:.2f} ppt); comparison-star systematics {1e3*selection['comparison_systematics']:.2f} ppt; time "
                  f"scale {scale_name}")
            if comparison_indices is not None:
                print("   Comparison stars chosen by the user; noise relative to photon noise (star: ratio; values well above the others "
                      f"indicate a variable or problematic star): {selection.get('comparison_noise_ratio', {})}")
        if plot:
            self.plot_differential_photometry(light_curve, image_scale=image_scale)
        return light_curve

    @staticmethod
    def _comparison_indices(comparison_stars, star_positions, reference_image, fwhm):
        """Indices of the measured stars closest to the comparison stars given by the user (pixel positions, coordinates or indices)."""
        indices = []
        for star in comparison_stars:
            if isinstance(star, (int, np.integer)):
                if not 0 < star < len(star_positions):
                    raise ValueError(f"Comparison star index {star} is not valid (1 to {len(star_positions) - 1}; 0 is the target).")
                indices.append(int(star))
                continue
            if isinstance(star, SkyCoord):
                if reference_image.wcs is None:
                    raise ValueError("Comparison stars given as coordinates need a WCS in the reference frame.")
                position = np.array(reference_image.wcs.celestial.world_to_pixel(star), dtype=float)
            else:
                position = np.asarray(star, dtype=float)
            distance = np.hypot(*(star_positions - position).T)
            distance[0] = np.inf                                   # never the target
            closest = int(np.argmin(distance))
            if distance[closest] > 2*fwhm:
                raise ValueError(f"No star within 2 FWHM of the comparison position {tuple(np.round(position, 1))} in the reference frame.")
            indices.append(closest)
        if len(set(indices)) < 1:
            raise ValueError("Give at least one comparison star.")
        return sorted(set(indices))

    def plot_differential_photometry(self, light_curve, image_scale="log", label_all_stars=True, percentile_interval=99.5):

        """
        The reference frame of a differential light curve, with the target (red circle), the comparison stars used (green circles,
        numbered) and the other measured stars (thin grey circles, numbered if label_all_stars=True). The numbers are the ones accepted by
        differential_photometry(comparison_stars=[...]) to choose the comparison stars manually.
        """

        image = light_curve.meta.get("reference_image")
        if image is None:
            raise ValueError("The light curve has no reference image (it was saved and read back?).")
        positions = np.asarray(light_curve.meta["star_positions"])
        comparisons = light_curve.meta["comparison_stars"]
        radius = light_curve.meta["aperture_radius_pixels"]
        inner_radius, outer_radius = light_curve.meta["sky_annulus_pixels"]
        background = image.background if image.background is not None else 0.0
        data = image.data - background
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            figure, axis = plt.subplots(figsize=(9.5, 9))
            shown = axis.imshow(data, origin="lower", cmap="magma", norm=_image_normalization(data, image_scale, percentile_interval),
                                interpolation="nearest")
            color_bar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
            color_bar.set_label(r"e$^-$ (" + image_scale + " scale)")
            for index, (x, y) in enumerate(positions):
                if index == 0 or index in comparisons:
                    continue
                axis.add_patch(plt.Circle((x, y), radius, fill=False, color="0.7", lw=0.6, alpha=0.7))
                if label_all_stars:
                    axis.annotate(str(index), (x, y), xytext=(radius*0.8, radius*0.8), textcoords="offset points", color="0.8", fontsize=7)
            for number, index in enumerate(comparisons):
                x, y = positions[index]
                axis.add_patch(plt.Circle((x, y), radius, fill=False, color="#2ECC71", lw=1.6,
                                          label="comparison stars" if number == 0 else None))
                axis.annotate(str(index), (x, y), xytext=(6, 6), textcoords="offset points", color="#2ECC71", fontsize=10, fontweight="bold")
            x, y = positions[0]
            axis.add_patch(plt.Circle((x, y), radius, fill=False, color="#E74C3C", lw=2.0, label="target"))
            axis.add_patch(plt.Circle((x, y), inner_radius, fill=False, color="#E74C3C", lw=0.7, ls="--", alpha=0.8, label="sky annulus"))
            axis.add_patch(plt.Circle((x, y), outer_radius, fill=False, color="#E74C3C", lw=0.7, ls="--", alpha=0.8))
            axis.set_xlim(-0.5, data.shape[1] - 0.5)
            axis.set_ylim(-0.5, data.shape[0] - 0.5)
            axis.set_aspect("equal")
            axis.set_title(f"{light_curve.meta.get('object', '')}: reference frame {image.name} ({light_curve.meta['comparison_choice']} "
                           f"comparison stars)", fontsize=12)
            axis.set_xlabel("x (pixels)")
            axis.set_ylabel("y (pixels)")
            axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.4, color="white")
            axis.tick_params(direction="in", top=True, right=True, color="0.6")
            axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
            plt.tight_layout()
            plt.show()

    def plot_light_curve(self, light_curve, number_of_points_per_bin=10, show_diagnostics=True, title=None):

        """Plot a differential light curve (with binned points) and, optionally, transparency, FWHM, sky and target position."""

        used = np.asarray(light_curve["used"])
        time = np.asarray(light_curve["time"])
        time_origin = np.floor(np.nanmin(time))
        hours = (time - time_origin)*24
        number_of_panels = 4 if show_diagnostics else 1
        figure, axes = plt.subplots(number_of_panels, 1, figsize=(9, 2.2 + 2.3*number_of_panels), sharex=True, squeeze=False,
                                    gridspec_kw={"height_ratios": [2.5] + [1]*(number_of_panels - 1)})
        axes = axes[:, 0]
        axes[0].errorbar(hours[used], light_curve["relative_flux"][used], light_curve["relative_flux_error"][used], fmt=".", ms=3, lw=0.5,
                         color="0.6", label="frames")
        binned_time, binned_flux, binned_error = aux_differential.bin_light_curve(
            hours[used], np.asarray(light_curve["relative_flux"])[used], number_of_points_per_bin,
            np.asarray(light_curve["relative_flux_error"])[used])
        axes[0].errorbar(binned_time, binned_flux, binned_error, fmt="o", ms=4, color="k", label=f"binned x{number_of_points_per_bin}")
        if np.any(~used):
            axes[0].plot(hours[~used], light_curve["relative_flux"][~used], "x", color="C3", ms=4, label="rejected")
            finite = np.isfinite(light_curve["relative_flux"][used])
            low, high = np.percentile(np.asarray(light_curve["relative_flux"][used])[finite], [0.5, 99.5])
            axes[0].set_ylim(low - 0.5*(high - low), high + 0.5*(high - low))
        axes[0].set_ylabel("Relative flux")
        axes[0].legend(fontsize=8)
        axes[0].set_title(title or f"{light_curve.meta.get('object', '')} ({light_curve.meta.get('band', '')}): "
                                   f"{len(light_curve.meta['comparison_stars'])} comparison stars, aperture "
                                   f"{light_curve.meta['aperture_radius_pixels']:.1f} px")
        if show_diagnostics:
            axes[1].plot(hours, light_curve["transparency"], ".", ms=3)
            axes[1].set_ylabel("Transparency")
            axes[2].plot(hours, light_curve["fwhm"], ".", ms=3, label="FWHM (px)")
            axes[2].set_ylabel("FWHM (px)")
            airmass_axis = axes[2].twinx()
            airmass_axis.plot(hours, light_curve["airmass"], "-", color="C1")
            airmass_axis.set_ylabel("Airmass", color="C1")
            axes[3].plot(hours, light_curve["x"] - np.nanmedian(light_curve["x"]), ".", ms=3, label="x")
            axes[3].plot(hours, light_curve["y"] - np.nanmedian(light_curve["y"]), ".", ms=3, label="y")
            axes[3].set_ylabel("Target drift (px)")
            axes[3].legend(fontsize=8)
        axes[-1].set_xlabel(f"Hours since {light_curve.meta['time_scale']} {time_origin:.0f}")
        plt.tight_layout()
        plt.show()

    def fit_transit(self, light_curve, period, mid_transit_time=None, planet_radius=0.1, scaled_semi_major_axis=10.0,
                    impact_parameter=0.3, limb_darkening=(0.4, 0.25), fit_limb_darkening=False, limb_darkening_prior_width=0.1,
                    fixed_parameters=None, baseline_polynomial_order=1, airmass_term=True, extra_regressors=None,
                    number_of_walkers=32, number_of_steps=4000, burn_in_fraction=0.4, seed=1, plot=True, verbose=True):

        """
        Fit a transit model to a differential light curve: transit (circular orbit, quadratic limb darkening) times a baseline.

        The free transit parameters are the mid-transit time, Rp/Rs, a/Rs and the impact parameter b (and the limb darkening if
        fit_limb_darkening=True, with Gaussian priors of width limb_darkening_prior_width around the given coefficients; use theoretical
        values, e.g. from Claret's tables, for the band). The baseline (a polynomial in time and, optionally, a term linear in the airmass
        and in extra regressors such as the FWHM) multiplies the transit; its coefficients are solved by linear least squares for every
        set of transit parameters, so they never compete with the transit parameters in the sampler.

        Steps: multi-start least squares; the errors are rescaled by sqrt(reduced chi2) and by the red-noise factor beta (time-averaging
        method), so that correlated noise is reflected in the uncertainties; MCMC (emcee) for the posteriors. The detection significance is
        given by the BIC difference between the models with and without the transit.

        Parameters
        ----------
        light_curve: Table
            Output of differential_photometry() (columns time, relative_flux, relative_flux_error, airmass, used).
        period: float
            Orbital period (days), fixed.
        mid_transit_time: float or None
            Initial guess of the mid-transit time (same time scale as the light curve). None: the time of the minimum of the smoothed
            light curve.
        planet_radius, scaled_semi_major_axis, impact_parameter: float
            Initial values of Rp/Rs, a/Rs and b.
        limb_darkening: tuple
            Quadratic limb-darkening coefficients (u1, u2).
        fit_limb_darkening: bool
            Fit the limb darkening (with priors) instead of fixing it.
        limb_darkening_prior_width: float
            Width of the Gaussian priors on u1 and u2 (only with fit_limb_darkening=True).
        fixed_parameters: dict or None
            Parameters to fix, e.g. {"scaled_semi_major_axis": 11.0, "impact_parameter": 0.4} (names: mid_transit_time, planet_radius,
            scaled_semi_major_axis, impact_parameter). Partial transits need a/Rs and b fixed (or with literature values).
        baseline_polynomial_order: int
            Order of the polynomial in time of the baseline.
        airmass_term: bool
            Include a term linear in the airmass (differential extinction between the target and the comparisons of different colors).
        extra_regressors: list of column names or None
            Columns of the light curve added linearly to the baseline (e.g. ["fwhm"]).
        number_of_walkers, number_of_steps, burn_in_fraction, seed: MCMC settings.
        plot: bool
            If True, call plot_transit_fit().
        verbose: bool
            If True, print the results.

        Returns
        -------
        fit: dict
            "parameters" {name: (median, upper error, lower error)} including the derived inclination, depth (Rp/Rs)^2 and duration T14
            (hours); "best_fit" (least-squares values); "chain" (posterior samples); "baseline", "transit_model" (arrays for the used
            points); "error_scale", "red_noise_factor", "delta_bic" (no transit - transit), "residual_rms"; and the data used.
        """

        import emcee
        used = np.asarray(light_curve["used"]) & np.isfinite(np.asarray(light_curve["relative_flux"], dtype=float))
        time = np.asarray(light_curve["time"], dtype=float)[used]
        flux = np.asarray(light_curve["relative_flux"], dtype=float)[used]
        flux_error = np.asarray(light_curve["relative_flux_error"], dtype=float)[used]
        airmass = np.asarray(light_curve["airmass"], dtype=float)[used] if "airmass" in light_curve.colnames else None
        if airmass_term and (airmass is None or not np.all(np.isfinite(airmass)) or np.ptp(airmass) == 0):
            airmass_term = False
        regressors = [np.asarray(light_curve[name], dtype=float)[used] for name in (extra_regressors or [])]
        design_matrix = aux_transit.baseline_design_matrix(time, airmass, baseline_polynomial_order, airmass_term, regressors)
        exposure_time = light_curve.meta.get("exposure_time")
        exposure_in_days = exposure_time/86400 if exposure_time else None
        if mid_transit_time is None:
            from scipy import ndimage
            smoothed = ndimage.median_filter(flux, max(5, len(flux)//15), mode="nearest")
            mid_transit_time = float(time[np.argmin(smoothed)])
        q1, q2 = aux_transit.kipping_from_limb_darkening(*limb_darkening)
        fixed = dict(fixed_parameters or {})
        priors = {}
        if fit_limb_darkening:
            # Gaussian priors on (u1, u2) translated approximately to (q1, q2) by numerical propagation
            samples = np.random.default_rng(0).normal(limb_darkening, limb_darkening_prior_width, (4000, 2))
            q_samples = np.array([aux_transit.kipping_from_limb_darkening(*sample) for sample in samples])
            q_samples = q_samples[np.all((q_samples > 0) & (q_samples < 1), axis=1)]
            priors["limb_darkening_q1"] = (q1, max(np.std(q_samples[:, 0]), 1e-3))
            priors["limb_darkening_q2"] = (q2, max(np.std(q_samples[:, 1]), 1e-3))
        else:
            fixed.setdefault("limb_darkening_q1", q1)
            fixed.setdefault("limb_darkening_q2", q2)
        initial = {"mid_transit_time": mid_transit_time, "planet_radius": planet_radius, "scaled_semi_major_axis": scaled_semi_major_axis,
                   "impact_parameter": impact_parameter, "limb_darkening_q1": q1, "limb_darkening_q2": q2}
        span = np.ptp(time)
        lower_limits = {"mid_transit_time": time.min() - 0.5*span, "planet_radius": 0.0, "scaled_semi_major_axis": 1.2,
                        "impact_parameter": 0.0, "limb_darkening_q1": 0.0, "limb_darkening_q2": 0.0}
        upper_limits = {"mid_transit_time": time.max() + 0.5*span, "planet_radius": 0.5, "scaled_semi_major_axis": 200.0,
                        "impact_parameter": 1.5, "limb_darkening_q1": 1.0, "limb_darkening_q2": 1.0}
        model = aux_transit.TransitModel(time, flux, flux_error, period, design_matrix, exposure_time=exposure_in_days, fixed=fixed,
                                         priors=priors)
        model.bounds = (np.array([lower_limits[name] for name in model.free_names]), np.array([upper_limits[name] for name in model.free_names]))
        # Multi-start least squares (mid-transit time and impact parameter are the parameters with multiple local minima)
        starting_points = []
        for time_offset in (np.linspace(-0.3, 0.3, 7)*span if "mid_transit_time" in model.free_names else [0.0]):
            for impact_start in ((0.1, 0.5, 0.8) if "impact_parameter" in model.free_names else (impact_parameter,)):
                point = dict(initial, mid_transit_time=mid_transit_time + time_offset, impact_parameter=impact_start)
                starting_points.append(np.array([point[name] for name in model.free_names]))
        best = aux_transit.multistart_least_squares(model, starting_points)
        best_parameters = model.full_parameters(best.x)
        best_shape = model.transit(best_parameters)
        best_baseline, _ = model.baseline(best_shape)
        number_of_free = len(model.free_names) + design_matrix.shape[1]
        reduced_chi2 = float(np.sum(((flux - best_baseline*best_shape)/flux_error)**2)/max(len(flux) - number_of_free, 1))
        residuals = flux/best_baseline - best_shape
        red_noise = aux_transit.red_noise_factor(residuals)
        error_scale = np.sqrt(max(reduced_chi2, 1e-12))*red_noise
        model.flux_error = flux_error*error_scale
        best = aux_transit.multistart_least_squares(model, [best.x])
        best_parameters = model.full_parameters(best.x)
        # BIC with and without the transit (with the rescaled errors)
        chi2_transit = float(np.sum(((flux - model.baseline(model.transit(best_parameters))[0]*model.transit(best_parameters))
                                     /model.flux_error)**2))
        null_baseline = design_matrix @ np.linalg.lstsq(design_matrix/model.flux_error[:, None], flux/model.flux_error, rcond=None)[0]
        chi2_null = float(np.sum(((flux - null_baseline)/model.flux_error)**2))
        delta_bic = (chi2_null + design_matrix.shape[1]*np.log(len(flux))) - (chi2_transit + number_of_free*np.log(len(flux)))
        # MCMC
        random_generator = np.random.default_rng(seed)
        scales = {"mid_transit_time": 5e-4, "planet_radius": 2e-3, "scaled_semi_major_axis": 0.1, "impact_parameter": 0.02,
                  "limb_darkening_q1": 0.02, "limb_darkening_q2": 0.02}
        start = best.x + random_generator.normal(0, [scales[name] for name in model.free_names], (number_of_walkers, len(model.free_names)))
        start = np.clip(start, model.bounds[0] + 1e-9, model.bounds[1] - 1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler = emcee.EnsembleSampler(number_of_walkers, len(model.free_names), model.log_probability)
            sampler.run_mcmc(start, number_of_steps, progress=False)
        chain = sampler.get_chain(discard=int(burn_in_fraction*number_of_steps), thin=5, flat=True)
        # Derived parameters
        samples = {name: chain[:, index] for index, name in enumerate(model.free_names)}
        for name, value in model.fixed.items():
            samples[name] = np.full(len(chain), value)
        samples["inclination"] = np.degrees(np.arccos(np.clip(samples["impact_parameter"]/samples["scaled_semi_major_axis"], -1, 1)))
        samples["depth"] = samples["planet_radius"]**2
        samples["duration_hours"] = 24*np.array([aux_transit.transit_duration(period, radius, semi_axis, inclination)
                                                 for radius, semi_axis, inclination in zip(samples["planet_radius"][::10],
                                                                                           samples["scaled_semi_major_axis"][::10],
                                                                                           samples["inclination"][::10])])
        u1_samples, u2_samples = aux_transit.limb_darkening_from_kipping(samples["limb_darkening_q1"], samples["limb_darkening_q2"])
        samples["limb_darkening_u1"], samples["limb_darkening_u2"] = u1_samples, u2_samples
        summary = {}
        for name, values in samples.items():
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            low, median, high = np.percentile(values, [16, 50, 84])
            summary[name] = (float(median), float(high - median), float(median - low))
        final_shape = model.transit(model.full_parameters(np.median(chain, axis=0)))
        final_baseline, baseline_coefficients = model.baseline(final_shape)
        fit = {"parameters": summary, "free_parameters": list(model.free_names), "fixed_parameters": dict(model.fixed),
               "best_fit": best_parameters, "chain": chain, "period": period, "time": time, "flux": flux, "flux_error": model.flux_error,
               "baseline": final_baseline, "transit_model": final_shape, "baseline_coefficients": baseline_coefficients,
               "error_scale": float(error_scale), "red_noise_factor": float(red_noise), "reduced_chi2_before_scaling": reduced_chi2,
               "delta_bic": float(delta_bic), "residual_rms": float(np.std(flux/final_baseline - final_shape)),
               "time_scale": light_curve.meta.get("time_scale"), "exposure_time": exposure_time,
               "acceptance_fraction": float(np.mean(sampler.acceptance_fraction))}
        if verbose:
            print(f"Transit fit: {len(flux)} points; errors scaled by {error_scale:.2f} (reduced chi2 {reduced_chi2:.2f}, red-noise factor "
                  f"{red_noise:.2f}); Delta BIC (no transit - transit) = {delta_bic:.1f}; residual rms {1e3*fit['residual_rms']:.2f} ppt")
            for name, label in (("mid_transit_time", f"Mid-transit time ({fit['time_scale']})"), ("planet_radius", "Rp/Rs"),
                                ("depth", "Depth (Rp/Rs)^2"), ("scaled_semi_major_axis", "a/Rs"), ("impact_parameter", "Impact parameter b"),
                                ("inclination", "Inclination (deg)"), ("duration_hours", "Duration T14 (h)"),
                                ("limb_darkening_u1", "u1"), ("limb_darkening_u2", "u2")):
                if name in summary:
                    median, upper, lower = summary[name]
                    fixed_text = " (fixed)" if name in model.fixed else ""
                    precision = 6 if name == "mid_transit_time" else 4
                    print(f"   {label:<34s} {median:.{precision}f} +{upper:.{precision}f} -{lower:.{precision}f}{fixed_text}")
        if plot:
            self.plot_transit_fit(fit)
        return fit

    def plot_transit_fit(self, fit, number_of_points_per_bin=10, title=None):
        """Detrended light curve (data / baseline) with the transit model and the residuals; binned points in black."""
        time, flux, flux_error = fit["time"], fit["flux"], fit["flux_error"]
        mid_transit_time = fit["parameters"]["mid_transit_time"][0]
        hours = (time - mid_transit_time)*24
        detrended = flux/fit["baseline"]
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        axes[0].errorbar(hours, detrended, flux_error/fit["baseline"], fmt=".", ms=3, lw=0.5, color="0.65", zorder=1)
        binned_time, binned_flux, binned_error = aux_differential.bin_light_curve(hours, detrended, number_of_points_per_bin,
                                                                                  flux_error/fit["baseline"])
        axes[0].errorbar(binned_time, binned_flux, binned_error, fmt="o", ms=4, color="k", zorder=3, label=f"binned x{number_of_points_per_bin}")
        fine_time = np.linspace(time.min(), time.max(), 2000)
        parameters = {name: values[0] for name, values in fit["parameters"].items()}
        u1, u2 = parameters["limb_darkening_u1"], parameters["limb_darkening_u2"]
        exposure_in_days = fit["exposure_time"]/86400 if fit.get("exposure_time") else None
        fine_model = aux_transit.transit_light_curve(fine_time, mid_transit_time, fit["period"], parameters["planet_radius"],
                                                     parameters["scaled_semi_major_axis"], parameters["inclination"], (u1, u2),
                                                     exposure_time=exposure_in_days)
        axes[0].plot((fine_time - mid_transit_time)*24, fine_model, color="C3", lw=2, zorder=4, label="transit model")
        axes[0].set_ylabel("Relative flux (detrended)")
        rp, rp_up, rp_low = fit["parameters"]["planet_radius"]
        axes[0].set_title(title or f"Rp/Rs = {rp:.4f} +{rp_up:.4f} -{rp_low:.4f};  $\\Delta$BIC = {fit['delta_bic']:.0f}")
        axes[0].legend(fontsize=8)
        residuals = (detrended - fit["transit_model"])*1e3
        axes[1].plot(hours, residuals, ".", ms=3, color="0.5")
        binned_time, binned_residuals, _ = aux_differential.bin_light_curve(hours, residuals, number_of_points_per_bin)
        axes[1].plot(binned_time, binned_residuals, "o", ms=4, color="k")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_ylabel("Residual (ppt)")
        axes[1].set_xlabel(f"Hours from mid-transit ({fit.get('time_scale')})")
        plt.tight_layout()
        plt.show()

    # ==================================================================================================================================
    # Blind search for transiting planets (the work is done in blind_search.py)
    # ==================================================================================================================================

    def blind_search(self, images, target_coordinates=None, target_magnitude=None, minimum_period=0.5, maximum_period=10.0,
                     transit_durations_hours=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0), sde_threshold=9.0, snr_threshold=8.0,
                     minimum_number_of_transits=2, maximum_depth=0.05, number_of_systematics=3, maximum_number_of_stars=300,
                     detection_threshold=5.0, number_of_cores=None, output_directory="blind_search_output", reuse_measurements=True,
                     plot=True, maximum_number_of_plotted_candidates=6, verbose=True):

        """
        Blind search for transiting planets around all the stars of a field observed on many nights, possibly by several observers with
        different telescopes, cameras, orientations and sites (e.g. a network of amateur astronomers). The work is done by the module
        easyspec.photometry.blind_search, in several processes in parallel (the photometry of the frames and the period search of the stars
        are independent tasks): see its documentation for the details of each step.

        Steps: star catalogue in sky coordinates (from the best frame of each instrument); aperture photometry of every star in every frame;
        light curves normalized per instrument and per frame, with the best aperture of each star; frames with clouds or bad seeing
        rejected; systematics common to many stars removed (SysRem); Box Least Squares period search of every star; vetting of the
        detections (number of transits, odd/even depths, secondary eclipse, sinusoidal variability, depth, periods shared by many stars).

        Parameters
        ----------
        images: str, Path, list of them, or list of Image
            Reduced frames (cleaning.save_fits_files()) or directories with them, of all nights and observers. Each frame needs a WCS
            (plate solution), DATE-OBS (UTC start) and EXPTIME, and preferably the site coordinates (SITELAT/SITELONG or LATITUDE/LONGITUD)
            for barycentric times. Frames from different instruments are recognized by OBSERVER, TELESCOP and INSTRUME (or their folder).
        target_coordinates: SkyCoord or (ra, dec) in degrees, optional
            The star at the centre of the field (e.g. the Gaia star the observers point at): it is marked in the results and used for the
            barycentric correction.
        target_magnitude: float, optional
            Magnitude of the target (e.g. Gaia G): the approximate magnitudes of all stars are computed relative to it.
        minimum_period, maximum_period: float
            Range of periods searched (days). The maximum is also limited to half the time span of the observations.
        transit_durations_hours: tuple
            Transit durations tried by the Box Least Squares search (hours).
        sde_threshold, snr_threshold, minimum_number_of_transits:
            Detection criteria: signal detection efficiency of the periodogram peak, depth / depth error, and number of transits with at
            least 4 points in transit.
        maximum_depth: float
            Deeper signals (relative flux) are flagged "too_deep" (most likely an eclipsing binary).
        number_of_systematics: int
            Number of SysRem components removed (2-5 is usual; too many can weaken real signals).
        maximum_number_of_stars: int
            Only the brightest stars of the catalogue are measured.
        detection_threshold: float
            Threshold (in background sigmas) of the star detection for the catalogue.
        number_of_cores: int or None
            Number of processes; default: all cores but one. Use 1 to run in this process only.
        output_directory: str or Path
            Where the results (blind_search_stars.ecsv, blind_search_candidates.ecsv, blind_search_frames.ecsv,
            blind_search_light_curves.fits) and the measurement cache (measurements/) are written.
        reuse_measurements: bool
            If True, frames measured in a previous run with the same catalogue are not measured again (useful when new nights are added).
        plot: bool
            If True, plot the summary (noise, signal-to-noise ratio vs period, field) and the vetting sheets of the candidates (the
            likely false positives can be plotted with result.plot_candidate(star_id)).
        maximum_number_of_plotted_candidates: int
            Maximum number of vetting sheets plotted.
        verbose: bool
            If True, print the progress and the detections.

        Returns
        -------
        result: blind_search.BlindSearchResult
            result.stars (table of all stars), result.candidates (detections), result.frames, result.time, result.relative_flux; and the
            methods light_curve(star_id, fold=False) (a table for fit_transit()), plot_summary(), plot_field(), plot_candidate(star_id),
            save(directory). A saved result is read back with blind_search.BlindSearchResult.load(output_directory).
        """

        from . import blind_search as blind_search_module
        result = blind_search_module.run_blind_search(
            images, target_coordinates=target_coordinates, target_magnitude=target_magnitude, minimum_period=minimum_period,
            maximum_period=maximum_period, transit_durations_hours=transit_durations_hours, sde_threshold=sde_threshold,
            snr_threshold=snr_threshold, minimum_number_of_transits=minimum_number_of_transits, maximum_depth=maximum_depth,
            number_of_systematics=number_of_systematics, maximum_number_of_stars=maximum_number_of_stars,
            detection_threshold=detection_threshold, number_of_cores=number_of_cores, output_directory=output_directory,
            reuse_measurements=reuse_measurements, verbose=verbose)
        if plot:
            result.plot_summary()
            candidates = result.candidates
            for star_id in list(candidates["id"][candidates["status"] == "candidate"])[:maximum_number_of_plotted_candidates]:
                result.plot_candidate(star_id)
        return result

    # ==================================================================================================================================
    # Plots and output
    # ==================================================================================================================================

    def plot_image(self, image, sources=None, photometry_table=None, standard_stars=None, image_scale="log", percentile_interval=99.5,
                   marker_radius=None, axis=None):

        """
        Show an image (background subtracted) with the detected sources or the stars of a photometry table (unflagged in blue, flagged in
        orange).

        Parameters
        ----------
        image: Image
            Image from load_images().
        sources: table or None
            Detections (output of find_stars()), drawn as circles.
        photometry_table: Table or None
            Output of aperture_photometry() or psf_photometry(); the stars are drawn with the color of their flags.
        standard_stars: Table, list of dicts, csv file or None
            Stars given by hand with positions x, y (e.g. the standard_stars of fit_standard_star_solution()), drawn as green circles
            labelled with their name or V magnitude: useful to check the identification of the standards.
        image_scale: "log" or "linear"
            Intensity scale of the display (default "log").
        percentile_interval: float
            Percentage of pixels inside the display range.
        marker_radius: float or None
            Radius (pixels) of the circles (default 2 FWHM).
        axis: matplotlib axis or None
            Axis to draw on.
        """

        show = axis is None
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            if axis is None:
                figure, axis = plt.subplots(figsize=(9, 8.5))
            else:
                figure = axis.figure
            data = image.data - (image.background if image.background is not None else 0)
            shown = axis.imshow(data, origin="lower", cmap="magma", norm=_image_normalization(data, image_scale, percentile_interval),
                                interpolation="nearest")
            color_bar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
            color_bar.set_label(r"e$^-$ (" + image_scale + " scale)")
            marker_size = (marker_radius or 2*(image.fwhm or 3.0))**2
            if photometry_table is not None:
                unflagged = np.asarray(photometry_table["flags"]) == 0
                axis.scatter(photometry_table["x"][unflagged], photometry_table["y"][unflagged], s=marker_size, facecolors="none",
                             edgecolors="#4FC3F7", lw=0.7, label=f"unflagged ({np.count_nonzero(unflagged)})")
                axis.scatter(photometry_table["x"][~unflagged], photometry_table["y"][~unflagged], s=marker_size, facecolors="none",
                             edgecolors="#FFB74D", lw=0.7, label=f"flagged ({np.count_nonzero(~unflagged)})")
                axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
            elif sources is not None and len(sources):
                axis.scatter(sources["x"], sources["y"], s=marker_size, facecolors="none", edgecolors="#4FC3F7", lw=0.7,
                             label=f"detections ({len(sources)})")
                axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
            if standard_stars is not None:
                stars = aux.read_catalogue(standard_stars, require_coordinates=False)
                axis.scatter(stars["x"], stars["y"], s=4*marker_size, facecolors="none", edgecolors="#69F0AE", lw=1.2,
                             label=f"standard stars ({len(stars)})")
                for index, star in enumerate(stars):
                    if not str(star["name"]).startswith("star_"):
                        label = str(star["name"])
                    elif "V" in stars.colnames:
                        label = f"V = {float(star['V']):.2f}"
                    else:
                        label = str(index + 1)
                    axis.annotate(label, (float(star["x"]), float(star["y"])), xytext=(8, 8), textcoords="offset points", color="#69F0AE",
                                  fontsize=9)
                axis.legend(loc="upper right", fontsize=9, framealpha=0.7)
            axis.set_title(image.name + (f" ({image.band})" if image.band else ""), fontsize=12)
            axis.set_xlabel("x (pixels)")
            axis.set_ylabel("y (pixels)")
            axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.4, color="white")
            axis.tick_params(direction="in", top=True, right=True, color="0.6")
            if show:
                plt.tight_layout()
                plt.show()
        return axis

    def plot_color_magnitude_diagram(self, catalogue, accepted_flags=0, maximum_color_error=0.2, isochrone=None, title=None, axis=None,
                                     color_by_density=True, show_error_bars=True, show_flagged=True):

        """
        Color-magnitude diagram of a two-band catalogue from calibrate_magnitudes().

        Parameters
        ----------
        catalogue: Table
            Output of calibrate_magnitudes() with two bands.
        accepted_flags: int
            Stars with (flags & ~accepted_flags) != 0 are drawn in grey (default: all flagged stars).
        maximum_color_error: float
            Stars with a color error above this are not shown.
        isochrone: dict or None
            Optional curve {"color": array, "magnitude": array, "label": str} (e.g. an isochrone shifted by the distance modulus and
            reddening).
        title: string
            Plot title.
        axis: matplotlib axis or None
            Axis to draw on.
        color_by_density: bool
            Color the unflagged stars by the local density of points in the diagram (it shows the sequences in crowded diagrams).
        show_error_bars: bool
            Show the median color and magnitude errors in bins of magnitude, on the right side of the diagram.
        show_flagged: bool
            Show the flagged stars (in light grey).
        """

        first_band, second_band = catalogue.meta["bands"]
        color_name = f"{first_band}-{second_band}"
        flags = np.asarray(catalogue["flags"])
        color = np.asarray(catalogue[color_name], dtype=float)
        magnitude = np.asarray(catalogue[second_band], dtype=float)
        color_error = np.asarray(catalogue[color_name + "_error"], dtype=float)
        magnitude_error = np.asarray(catalogue[second_band + "_error"], dtype=float)
        shown = np.isfinite(color) & np.isfinite(magnitude) & (color_error < maximum_color_error)
        unflagged = shown & ((flags & ~accepted_flags) == 0)
        flagged = shown & ~unflagged
        show = axis is None
        with plt.rc_context({"axes.linewidth": 0.8, "font.size": 11}):
            if axis is None:
                figure, axis = plt.subplots(figsize=(6.5, 8))
            if show_flagged and np.any(flagged):
                axis.scatter(color[flagged], magnitude[flagged], s=5, color="0.75", alpha=0.6, linewidths=0, zorder=1,
                             label=f"flagged ({np.count_nonzero(flagged)})")
            if np.any(unflagged):
                if color_by_density and np.count_nonzero(unflagged) > 50:
                    # Density of points (Gaussian kernel in the color-magnitude plane, from a fine 2D histogram)
                    from scipy import ndimage
                    histogram, color_edges, magnitude_edges = np.histogram2d(color[unflagged], magnitude[unflagged], bins=(120, 160))
                    histogram = ndimage.gaussian_filter(histogram, 1.5)
                    color_index = np.clip(np.searchsorted(color_edges, color[unflagged]) - 1, 0, histogram.shape[0] - 1)
                    magnitude_index = np.clip(np.searchsorted(magnitude_edges, magnitude[unflagged]) - 1, 0, histogram.shape[1] - 1)
                    density = histogram[color_index, magnitude_index]
                    order = np.argsort(density)
                    axis.scatter(color[unflagged][order], magnitude[unflagged][order], c=np.log10(density[order] + 1), cmap="viridis",
                                 s=7, linewidths=0, zorder=2, label=f"unflagged ({np.count_nonzero(unflagged)})")
                else:
                    axis.scatter(color[unflagged], magnitude[unflagged], s=7, color="#2C3E50", linewidths=0, zorder=2,
                                 label=f"unflagged ({np.count_nonzero(unflagged)})")
            if isochrone is not None:
                axis.plot(isochrone["color"], isochrone["magnitude"], color="#E74C3C", lw=1.8, zorder=3, label=isochrone.get("label"))
            if show_error_bars and np.count_nonzero(unflagged) > 10:
                # Typical errors in bins of 1 magnitude, drawn at the right edge of the diagram
                low, high = np.floor(np.nanmin(magnitude[unflagged])), np.ceil(np.nanmax(magnitude[unflagged]))
                color_range = np.nanpercentile(color[unflagged], [0.5, 99.5])
                bar_position = color_range[1] + 0.12*np.ptp(color_range)
                error_bar_label = "median errors (per magnitude)"
                for bin_start in np.arange(low, high, 1.0):
                    in_bin = unflagged & (magnitude >= bin_start) & (magnitude < bin_start + 1)
                    if np.count_nonzero(in_bin) >= 3:
                        axis.errorbar(bar_position, bin_start + 0.5, xerr=np.median(color_error[in_bin]),
                                      yerr=np.median(magnitude_error[in_bin]), fmt="none", ecolor="#C0392B", elinewidth=1.2, capsize=2,
                                      zorder=4, label=error_bar_label)
                        error_bar_label = None           # only the first bar goes to the legend
            # Axis limits from the bulk of the unflagged stars (outliers would squeeze the sequences)
            reference = unflagged if np.count_nonzero(unflagged) > 10 else shown
            if np.count_nonzero(reference) > 10:
                color_low, color_high = np.nanpercentile(color[reference], [0.5, 99.5])
                margin = 0.12*(color_high - color_low)
                axis.set_xlim(color_low - margin, color_high + 2*margin)
                magnitude_low, magnitude_high = np.nanmin(magnitude[shown]), np.nanmax(magnitude[shown])
                axis.set_ylim(magnitude_high + 0.3, magnitude_low - 0.3)
            else:
                axis.invert_yaxis()
            axis.set_xlabel(f"${first_band} - {second_band}$")
            axis.set_ylabel(f"${second_band}$")
            axis.minorticks_on()
            axis.tick_params(which="both", direction="in", top=True, right=True)
            axis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.5)
            legend = axis.legend(loc="upper right", fontsize=9, markerscale=2.5, frameon=True, framealpha=0.85)
            legend.get_frame().set_edgecolor("0.8")
            text = title or catalogue.meta.get("object") or ""
            method = catalogue.meta.get("method")
            method_name = {"psf": "PSF", "aperture": "aperture"}.get(method, method)
            axis.set_title(f"{text}" + (f" ({method_name} photometry)" if method and title is None else ""), fontsize=12)
            if show:
                plt.tight_layout()
                plt.show()
        return axis

    @staticmethod
    def save_table(table, file_name, overwrite=True):
        """Save a table (csv, ecsv or fits, from the extension). Non-serializable metadata (WCS, arrays, functions) is dropped."""
        table = table.copy()
        table.meta = {key: value for key, value in table.meta.items()
                      if key not in ("wcs", "measurements", "star_positions", "isophotes", "reference_image")
                      and isinstance(value, (str, int, float, tuple, list, dict, type(None)))}
        file_name = Path(file_name)
        table_format = {".csv": "ascii.csv", ".ecsv": "ascii.ecsv", ".fits": "fits", ".dat": "ascii.fixed_width"}.get(file_name.suffix.lower())
        table.write(file_name, format=table_format, overwrite=overwrite)
        return file_name
