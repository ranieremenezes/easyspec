from astropy.io import fits
import numpy as np
import scipy
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from functools import wraps
from scipy import interpolate
from scipy.integrate import trapezoid
from easyspec.cleaning import cleaning as _cleaning
from easyspec import data_io
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter
from astropy import units as u
from scipy.signal import medfilt
import scipy.ndimage, scipy.optimize, scipy.signal
from astropy.stats import sigma_clipped_stats
from dust_extinction.parameter_averages import F99
from easyspec.extraction import optimal_extraction
from easyspec.extraction import line_identification
from easyspec.extraction import flux_calibration
from easyspec.extraction import telluric
from astropy.table import Table
import warnings
import copy

libpath = Path(__file__).parent.resolve() / Path("airmass")
libpath_std = Path(__file__).parent.resolve() / Path("standards")

easyspec_extraction_version = "1.2.0"

# Matplotlib settings used by the easyspec plots. They are applied only while an easyspec function is running (see _plot_style), so the
# user's own matplotlib settings are never changed.
_PLOT_RC = {'font.size': 12}


def _plot_style(function):

    """Decorator: run 'function' with the easyspec plot settings (_PLOT_RC), restoring the user's matplotlib settings afterwards."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_PLOT_RC):
            return function(*args, **kwargs)
    return wrapper


def _look_in_header(file_name, keyword):

    """Return the value of 'keyword' from the first HDU of the fits file that contains it. Raise an error if no HDU contains it."""

    with fits.open(file_name) as hdul:
        value = data_io._find_keyword(hdul, keyword)
    if value is None:
        raise RuntimeError(f"No entry called '{keyword}' was found in the header of {Path(file_name).name}. Please check the header "
                           "keyword, or give the value manually (e.g. with exposure_target=... or airmass_target=...).")
    return value


def _plot_image(image_data, title):

    """Plot an image in logarithmic scale (without blank pixels where the image is zero or negative, see cleaning._log_display)."""

    image_shape = np.shape(image_data)
    plt.figure(figsize=(12, 12*image_shape[0]/image_shape[1]))
    plt.title(title)
    ax = plt.gca()
    im = ax.imshow(_cleaning._log_display(image_data), origin='lower', cmap='gray')
    cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.08)
    plt.colorbar(im, cax=cax, label="Counts in log scale")
    plt.show()


class extraction:

    """This class contains all the functions necessary to perform the spectral extraction from a cleaned/stacked image."""

    def __init__(self):
        # Print the current version of easyspec-extraction
        print("easyspec-extraction version: ",easyspec_extraction_version)

        # Attributes filled by the other functions of this class. They are created here, so that the functions can be called in any
        # order without raising AttributeError.
        self.target_spec_data = None
        self.target_variance = None         # Variance of the target image ("VAR" extension), filled by import_data()
        self.target_mask = None             # Data quality mask of the target image ("MASK" extension), filled by import_data()
        self.std_star_spec_data = None
        self.std_star_variance = None
        self.std_star_mask = None
        self.target_name = None
        self.exposure_target = None
        self.airmass_target = None
        self.exposure_std_star = None
        self.airmass_std_star = None
        self.image_shape = None
        self.aspect_ratio = None
        self.wavelengths_fit_rms_list = None  # To be filled with the function extraction.wavelength_calibration()
        self.spec_systematic_error_list = None  # To be filled with the function extraction.extracting()
        self.extraction_weights = None  # To be filled with the function extraction.extracting()

    def find_nearest(self,array, value):

        """For a monotonically increasing/decreasing array, this function finds the index of the element nearest to the given value

        Parameters
        ----------
        array: numpy array (float)
            Monotonically increasing/decreasing array.

        value: float
            Value to look for in the array.

        Returns
        -------
        idx: int
            The index corresponding to the closest element to the given value.

        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @_plot_style
    def import_data(self, target_spec_file, target_name, exposure_target = None, exposure_header_entry="AVEXP", airmass_target = None, airmass_header_entry="AVAIRMAS", std_star_spec_file = None, exposure_std_star = None, airmass_std_star = None, plot = True):

        """
        This function opens the data file (i.e. fits image with one or more spectra) and reads the exposure time and airmass from its header.
        Optionally, the user can manually give the exposure time and the airmass. If you want to calibrate your spectrum in flux, please also
        give the standard star spectral fits file.

        If the files were produced by easyspec.cleaning(), their variance ("VAR" extension) and data quality mask ("MASK" extension) are also
        read and stored in the attributes self.target_variance, self.target_mask, self.std_star_variance and self.std_star_mask.

        Parameters
        ----------
        target_spec_file: string
            The path to the data '.fits' or '.fit' file containing the target spectrum.
        target_name: string
            The name of the target to be used in subsequent plots.
        exposure_target: float
            Optional. You can manually pass the exposure time (in seconds) of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'exposure_header_entry'.
        exposure_header_entry: string
            Header entry with the exposure time. Used only if 'exposure_target' (or 'exposure_std_star') is None. This entry is used for the target and standard star images.
        airmass_target: float
            Optional. You can manually pass the average airmass of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'airmass_header_entry'.
        airmass_header_entry: string
            Header entry with the average airmass. Used only if 'airmass_target' (or 'airmass_std_star') is None. This entry is used for the target and standard star images.
            If None, the airmass is not read (and must be given manually before the extinction correction).
        std_star_spec_file: string
            Optional. The path to the data '.fits' or '.fit' file containing the standard star spectrum.
        exposure_std_star: float
            Optional. You can manually pass the exposure time (in seconds) of your standard star observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'exposure_header_entry'.
        airmass_std_star: float
            Optional. You can manually pass the average airmass of your standard star observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'airmass_header_entry'.
        plot: boolean
            If True, easyspec will plot the input spectra.


        Returns
        -------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        std_star_spec_data: numpy.ndarray (float)
            Matrix containing the standard star spectral image (only if the standard star spectral file is given by the user).
        """

        # Main target:
        self.target_spec_data, self.target_variance, self.target_mask, self.target_header = data_io.read_reduced_frame(target_spec_file, estimate_variance=False)
        self.target_name = target_name
        if exposure_target is None:
            self.exposure_target = _look_in_header(target_spec_file,exposure_header_entry)
        else:
            self.exposure_target = exposure_target
        if airmass_target is not None:
            self.airmass_target = airmass_target
        elif airmass_header_entry is not None:
            self.airmass_target = _look_in_header(target_spec_file,airmass_header_entry)

        self.image_shape = np.shape(self.target_spec_data)
        self.aspect_ratio = self.image_shape[0]/self.image_shape[1]

        # Standard star:
        if std_star_spec_file is not None:
            self.std_star_spec_data, self.std_star_variance, self.std_star_mask, self.std_star_header = data_io.read_reduced_frame(std_star_spec_file, estimate_variance=False)
            if exposure_std_star is not None:
                self.exposure_std_star = exposure_std_star
            else:
                self.exposure_std_star = _look_in_header(std_star_spec_file,exposure_header_entry)
            if airmass_std_star is not None:
                self.airmass_std_star = airmass_std_star
            elif airmass_header_entry is not None:
                self.airmass_std_star = _look_in_header(std_star_spec_file,airmass_header_entry)
            if np.shape(self.std_star_spec_data) != self.image_shape:
                warnings.warn(f"The standard star image has shape {np.shape(self.std_star_spec_data)}, but the target image has shape "
                              f"{self.image_shape}. They should have been reduced with the same trim (and the same lamp).")

        missing_variance = [name for name, variance in (("target", self.target_variance), ("standard star", self.std_star_variance))
                            if variance is None and (name == "target" or std_star_spec_file is not None)]
        if missing_variance:
            print("Note: no variance ('VAR' extension) found for the " + " and ".join(missing_variance) + " image(s). To propagate the "
                  "uncertainties, reduce the data with easyspec.cleaning() using compute_variance() and master(..., variance=variance).")

        if plot:
            _plot_image(self.target_spec_data, target_name)
            if std_star_spec_file is not None:
                _plot_image(self.std_star_spec_data, "Standard star")

        if std_star_spec_file is not None:
            print("Returning both target and std star images...")
            return self.target_spec_data, self.std_star_spec_data
        else:
            print("No standard star file was given. Returning only the target image.")
            return self.target_spec_data
    
    # ==================================================================================================================================
    # Tracing
    # ==================================================================================================================================

    def _image_quality_arrays(self, image):

        """
        Return (variance, mask) for the image given to tracing()/extracting(): the variance and data quality mask read by import_data() if
        'image' is the target or standard star image loaded there, or (None, None) otherwise.
        """

        if image is self.target_spec_data:
            return self.target_variance, self.target_mask
        if image is self.std_star_spec_data:
            return self.std_star_variance, self.std_star_mask
        return None, None

    @staticmethod
    def _gaussian_with_line(y, amplitude, center, sigma, b0, b1):

        """Gaussian profile on top of a linear background, used to measure the center of the trace in each bin of columns."""

        return amplitude*np.exp(-0.5*((y - center)/sigma)**2) + b0 + b1*(y - center)

    def _bin_columns(self, image, good, variance, bin_size, x_min, x_max):

        """
        Combine the columns of 'image' between x_min and x_max (inclusive) in bins of 'bin_size' columns, ignoring the bad pixels (good = False).

        The bins are combined with a median, which is robust against residual cosmic rays and hot pixels. Its variance is (pi/2) times the
        variance of the mean (for Gaussian noise), and is computed from 'variance' if it is available.

        Returns
        -------
        x_centers: array
            Mean x position of each bin.
        binned: array, shape (ny, n_bins)
            Median of each row in each bin (NaN where all pixels are bad).
        binned_variance: array or None
            Variance of 'binned' (None if 'variance' is None).
        """

        edges = np.arange(x_min, x_max + 1, bin_size)
        if x_max + 1 - edges[-1] < bin_size/2 and len(edges) > 1:   # Merge a short last bin with the previous one
            edges = edges[:-1]
        edges = np.append(edges, x_max + 1)
        ny = image.shape[0]
        x_centers, binned, binned_variance = [], [], []
        masked_image = np.where(good, image, np.nan)
        for x1, x2 in zip(edges[:-1], edges[1:]):
            block = masked_image[:, x1:x2]
            n_valid = np.sum(np.isfinite(block), axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # All-NaN rows
                binned.append(np.nanmedian(block, axis=1))
                if variance is not None:
                    mean_variance = np.nanmean(np.where(good[:, x1:x2], variance[:, x1:x2], np.nan), axis=1)
                    binned_variance.append(np.where(n_valid > 0, (np.pi/2)*mean_variance/np.maximum(n_valid, 1), np.nan))
            x_centers.append(0.5*(x1 + x2 - 1))
        binned = np.array(binned).T
        binned_variance = np.array(binned_variance).T if variance is not None else None
        return np.array(x_centers), binned, binned_variance

    def _spatial_profile(self, image, good, x_min, x_max):

        """
        Robust spatial profile of the image between the columns x_min and x_max (inclusive): for each row, the sigma-clipped mean of its
        pixels (insensitive to cosmic rays, hot pixels and emission lines), minus the median background level of the profile.

        Returns the profile and its noise (per row), estimated from the scatter of the pixels of each row.
        """

        block = np.where(good, image, np.nan)[:, x_min:x_max + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, _, std = sigma_clipped_stats(block, axis=1, sigma=3.0, maxiters=5)
            n_valid = np.sum(np.isfinite(block), axis=1)
        mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        profile = np.where(np.isfinite(mean), mean, np.nanmedian(mean))
        profile = profile - np.median(profile)
        noise = std/np.sqrt(np.maximum(n_valid, 1))
        noise = np.where(np.isfinite(noise) & (noise > 0), noise, np.nanmedian(noise[np.isfinite(noise) & (noise > 0)]) if np.any(np.isfinite(noise) & (noise > 0)) else 1.0)
        return profile, noise

    def _detect_traces(self, image, good, method, x_min, x_max, peak_height, distance, detection_sigma, y_pixel_range):

        """
        Initial y position of the trace(s), measured on the robust spatial profile (see _spatial_profile) of a block of columns. The block
        is narrow enough (1/5 of the x range, at least 50 columns) for tilted or curved traces not to be smeared out. The central block is
        used first and, if no spectrum is detected there, the other blocks are tried, from the center outwards.

        The profile is smoothed with a Gaussian of 1.5 pixels (a matched filter for the spatial profile of a spectrum, which suppresses
        single-row noise peaks), and a running median of the profile is subtracted as the local background. A peak is accepted if its height
        above this background is larger than 'peak_height' (in counts) or, if peak_height is None, larger than 'detection_sigma' times the
        (robustly estimated) noise of the smoothed profile. Its prominence must also be at least half of this threshold, so that bumps in
        the wings of a brighter spectrum are not taken as new spectra. For the single-trace methods, only the most significant peak is
        returned; for method = "multi", all peaks separated by at least 'distance' pixels.

        Returns (y_positions, x_block_center).
        """

        ny = image.shape[0]
        width = max(50, (x_max - x_min + 1)//5)
        n_blocks = max(1, (x_max - x_min + 1)//width)
        starts = x_min + np.arange(n_blocks)*((x_max - x_min + 1 - width)//max(n_blocks - 1, 1)) if n_blocks > 1 else np.array([x_min])
        middle = 0.5*(x_min + x_max)
        starts = sorted(starts, key=lambda s: abs(s + width/2 - middle))   # From the center outwards
        edge = min(int(y_pixel_range), ny//4)   # Peaks are searched only where a full window of +-y_pixel_range fits inside the image

        for start in starts:
            x1, x2 = int(start), int(min(x_max, start + width - 1))
            profile, _ = self._spatial_profile(image, good, x1, x2)
            smoothed = scipy.ndimage.gaussian_filter1d(profile, 1.5, mode="nearest")
            local_background = scipy.ndimage.median_filter(smoothed, size=max(31, 4*int(y_pixel_range) + 1), mode="nearest")
            excess = smoothed - local_background
            noise = 1.4826*np.median(np.abs(excess - np.median(excess)))
            if not noise > 0:
                noise = np.std(excess) if np.std(excess) > 0 else 1.0
            threshold = peak_height if peak_height is not None else detection_sigma*noise
            search = excess.copy()
            search[:edge] = -np.inf
            search[ny - edge:] = -np.inf
            peaks, properties = scipy.signal.find_peaks(search, height=threshold, prominence=0.5*threshold, distance=max(1, int(distance)))
            score = properties["peak_heights"] if len(peaks) else np.array([])
            if len(peaks) == 0:
                continue
            if method != "multi":
                peaks = peaks[[np.argmax(score)]]
            refined = []
            for p in peaks:   # Sub-pixel refinement of each peak with a parabola through the 3 highest pixels
                a, b, c = profile[p - 1], profile[p], profile[p + 1]
                denominator = a - 2*b + c
                refined.append(p + (float(np.clip(0.5*(a - c)/denominator, -0.5, 0.5)) if denominator < 0 else 0.0))
            return np.array(refined), 0.5*(x1 + x2)

        threshold = f"{peak_height} counts" if peak_height is not None else f"{detection_sigma} times its noise"
        raise RuntimeError(f"No spectrum was detected: the spatial profile of the image has no peak above {threshold}. Try a lower "
                           "'peak_height' (or peak_height=None with a lower detection_sigma), or check 'xlims'.")

    def _measure_center(self, y, values, values_variance, predicted, half_window, trace_half_width, method):

        """
        Measure the center of the trace in one bin of columns, within +-half_window pixels of the predicted position.

        Returns (center, center_error, sigma, snr), or None if the measurement failed or is not significant. 'sigma' is the Gaussian width of
        the spatial profile (only for method = "profile"; NaN otherwise).
        """

        inside = (y >= predicted - half_window) & (y <= predicted + half_window) & np.isfinite(values)
        if values_variance is not None:
            inside &= np.isfinite(values_variance) & (values_variance > 0)
        if np.count_nonzero(inside) < 7:
            return None
        yy, vv = y[inside], values[inside]
        ee = np.sqrt(values_variance[inside]) if values_variance is not None else None

        # Local linear background from the outer parts of the window, and the noise around it:
        outer = np.abs(yy - predicted) > max(trace_half_width, 0.5*half_window)
        if np.count_nonzero(outer) >= 4:
            background_fit = np.polynomial.Polynomial.fit(yy[outer], vv[outer], 1)
            background = background_fit(yy)
            noise = 1.4826*np.median(np.abs(vv[outer] - background[outer]))
        else:
            background = np.full(len(yy), np.median(vv))
            noise = 1.4826*np.median(np.abs(vv - np.median(vv)))
        if ee is not None:
            noise = np.median(ee)
        if not noise > 0:
            return None
        signal = vv - background
        peak_index = np.argmax(signal)
        snr = signal[peak_index]/noise
        if not snr > 0:
            return None

        if method == "argmax":
            p = peak_index
            if 0 < p < len(signal) - 1:   # Sub-pixel refinement with a parabola through the 3 highest pixels
                a, b, c = signal[p - 1], signal[p], signal[p + 1]
                denominator = a - 2*b + c
                center = yy[p] + (0.5*(a - c)/denominator if denominator < 0 else 0.0)
            else:
                center = float(yy[p])
            # Uncertainty: half a pixel scaled down by the S/N of the peak (and never below 0.05 pixels)
            return center, max(0.5/snr, 0.05), np.nan, snr

        if method == "moments":
            core = np.abs(yy - yy[peak_index]) <= trace_half_width
            weights = np.clip(signal[core], 0, None)
            if weights.sum() <= 0:
                return None
            center = np.sum(weights*yy[core])/weights.sum()
            # Error of the first moment: sum_i ((y_i - center)/W)^2 * sigma_i^2
            pixel_noise = ee[core] if ee is not None else np.full(np.count_nonzero(core), noise)
            center_error = np.sqrt(np.sum(((yy[core] - center)/weights.sum())**2*pixel_noise**2))
            return center, max(center_error, 0.01), np.nan, snr

        # method == "profile": Gaussian + linear background fit
        amplitude0 = max(signal[peak_index], noise)
        p0 = [amplitude0, yy[peak_index], max(trace_half_width/3, 1.0), background[peak_index], 0.0]
        lower = [0.0, yy.min(), 0.3, -np.inf, -np.inf]
        upper = [np.inf, yy.max(), max(half_window, 1.0), np.inf, np.inf]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = scipy.optimize.curve_fit(self._gaussian_with_line, yy, vv, p0=p0, bounds=(lower, upper),
                                                      sigma=ee, absolute_sigma=ee is not None, maxfev=2000)
        except (RuntimeError, ValueError):
            return None
        amplitude, center, sigma = popt[0], popt[1], popt[2]
        if not np.all(np.isfinite(pcov)):
            return None
        center_error = np.sqrt(pcov[1, 1])
        amplitude_error = np.sqrt(pcov[0, 0])
        # Reject fits stuck at the bounds or not significant:
        if (abs(center - yy.min()) < 0.5 or abs(center - yy.max()) < 0.5 or sigma <= 0.31 or sigma >= 0.99*upper[2]
                or not amplitude > 3*amplitude_error or not center_error > 0):
            return None
        return center, center_error, sigma, amplitude/amplitude_error

    def _follow_trace(self, x_bins, binned, binned_variance, y0, x0, half_window, trace_half_width, method, min_snr):

        """
        Measure the center of one trace in every bin of columns. The trace is followed from the bin closest to x0 (where its initial
        position y0 was detected) towards both ends of the detector, and the window where the center is searched is always placed at the
        position predicted from the bins already measured (so that curved and tilted traces are followed, and a bin with no signal does not
        make the trace jump to another source).

        Returns arrays (centers, errors, sigmas, snrs), with NaN in the bins where the measurement failed.
        """

        ny, n_bins = binned.shape
        y = np.arange(ny, dtype=float)
        centers, errors, sigmas, snrs = (np.full(n_bins, np.nan) for _ in range(4))

        start = int(np.argmin(np.abs(x_bins - x0)))   # Starting bin: where the initial position y0 was measured

        def predict(index, direction):
            # Position predicted from the (up to) 5 nearest successful measurements on the side already traced
            if direction > 0:
                done = np.nonzero(np.isfinite(centers[:index]))[0][-5:]
            else:
                done = np.nonzero(np.isfinite(centers[index + 1:]))[0][:5] + index + 1
            if len(done) == 0:
                return y0
            if len(done) == 1:
                return centers[done[0]]
            slope, intercept = np.polyfit(x_bins[done], centers[done], 1)
            return slope*x_bins[index] + intercept

        for direction, indexes in ((+1, range(start, n_bins)), (-1, range(start - 1, -1, -1))):
            for index in indexes:
                predicted = predict(index, direction) if not (direction > 0 and index == start) else y0
                variance_column = binned_variance[:, index] if binned_variance is not None else None
                result = self._measure_center(y, binned[:, index], variance_column, predicted, half_window, trace_half_width, method)
                if result is None or result[3] < min_snr or abs(result[0] - predicted) > half_window:
                    continue
                centers[index], errors[index], sigmas[index], snrs[index] = result
        return centers, errors, sigmas, snrs

    def _fit_trace(self, x_bins, centers, errors, poly_order, clip_sigma, max_iterations, nx):

        """
        Weighted (1/error^2) polynomial fit of the measured trace centers, with iterative sigma clipping. The polynomial is an astropy
        Polynomial1D defined on the domain [0, nx-1] (mapped to [-1, 1]), which keeps the fit well conditioned for any order.

        Returns (model, used, rms), where 'used' flags the bins kept in the final fit and 'rms' is the weighted RMS of their residuals (pixels).
        """

        valid = np.isfinite(centers) & np.isfinite(errors) & (errors > 0)
        n_valid = np.count_nonzero(valid)
        order = poly_order
        if n_valid < order + 2:
            order = max(0, n_valid - 2)
            if n_valid < 2:
                raise RuntimeError(f"Only {n_valid} bin(s) of columns had a significant detection of the trace, which is not enough to fit it. "
                                   "Try a larger 'bin_size', a lower 'min_snr', or check 'xlims' and 'y_pixel_range'.")
            warnings.warn(f"Only {n_valid} bins of columns had a significant detection of the trace, so its polynomial order was reduced "
                          f"from {poly_order} to {order}.")

        # A floor of 0.02 pixels in the errors avoids giving an excessive weight to a few very bright bins:
        weights = 1.0/np.sqrt(errors**2 + 0.02**2)
        used = valid.copy()
        fitter = LinearLSQFitter()
        model = None
        for _ in range(max_iterations):
            template = Polynomial1D(degree=order, domain=[0, nx - 1], window=[-1, 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = fitter(template, x_bins[used], centers[used], weights=weights[used])
            normalized_residuals = (centers - model(x_bins))*weights
            scatter = 1.4826*np.median(np.abs(normalized_residuals[used]))
            if not scatter > 0:
                break
            new_used = valid & (np.abs(normalized_residuals) <= clip_sigma*max(scatter, 1.0))
            if np.count_nonzero(new_used) < order + 2 or np.array_equal(new_used, used):
                break
            used = new_used
        residuals = centers[used] - model(x_bins[used])
        rms = float(np.sqrt(np.average(residuals**2, weights=weights[used]**2))) if np.any(used) else np.nan
        return model, used, rms

    def _repeat_trace(self, fitted_polymodel_list, trace_info, repeat_trace_at):

        """
        Duplicate traces shifted along the y axis (see the parameter 'repeat_trace_at' of tracing()). The new traces are copies of the
        original polynomials with a constant offset. The final list is sorted from the bottom to the top of the image.
        """

        if not isinstance(repeat_trace_at, dict):
            raise TypeError("'repeat_trace_at' must be a dictionary like {'spec_0': 75, 'spec_1': -50}.")
        n_original = len(fitted_polymodel_list)
        for spec_key, shift in repeat_trace_at.items():
            try:
                index = int(str(spec_key).split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid key '{spec_key}' in 'repeat_trace_at'. The keys must be 'spec_0', 'spec_1', etc.")
            if not 0 <= index < n_original:
                raise ValueError(f"'repeat_trace_at' refers to {spec_key}, but only {n_original} trace(s) were found (spec_0 to spec_{n_original - 1}).")
            new_trace = copy.deepcopy(fitted_polymodel_list[index])
            new_trace.c0 = new_trace.c0.value + float(shift)   # In the [-1, 1] window, c0 is a constant offset of the whole trace
            fitted_polymodel_list.append(new_trace)
            info = copy.deepcopy(trace_info[index])
            info["centers"] = info["centers"] + float(shift)
            info["repeated_from"] = spec_key
            info["shift"] = float(shift)
            trace_info.append(info)

        x_middle = np.array([info["x_bins"].mean() for info in trace_info])
        mean_position = [model(x) for model, x in zip(fitted_polymodel_list, x_middle)]
        order = np.argsort(mean_position)
        return [fitted_polymodel_list[i] for i in order], [trace_info[i] for i in order]

    @_plot_style
    def tracing(self, target_spec_data, method = "profile", y_pixel_range = 15, xlims = None, poly_order = 2, trace_half_width = 7, peak_height = None,
                distance = 50, Number_of_slices = None, peak_dispersion_limit = None, repeat_trace_at = None, main_plot = True, plot_residuals = True,
                bin_size = None, min_snr = 3.0, detection_sigma = 5.0, clip_sigma = 3.0, max_iterations = 10):

        """
        This function detects the spectra in an image and fits a polynomial to the trace (or spine) of each one of them.

        How it works:

        1. The initial y position of each spectrum is found as a peak of the robust spatial profile of the image (the median of each row
           between the limits given by 'xlims'), which is insensitive to cosmic rays and to emission lines.
        2. The columns are combined (with a median) in bins of 'bin_size' columns, which increases the signal-to-noise ratio, and the
           center of the trace is measured in each bin within +-y_pixel_range pixels of its predicted position. The trace is followed from
           its brightest part towards both ends of the detector, so curved and tilted traces are followed, and bins without a significant
           detection are skipped instead of making the trace jump to another source.
        3. A polynomial is fitted to the centers, weighted by their uncertainties, with iterative sigma clipping.

        If the image is the target or standard star image loaded with import_data() and it has a data quality mask (files produced by
        easyspec.cleaning()), pixels flagged in the mask (cosmic rays, non-finite values, no data after the alignment) are ignored, and the
        variance of the image is used to weight the measurements.

        Parameters
        ----------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target (or standard star) spectral image.
        method: string
            How the center of the trace is measured in each bin of columns:
            - "profile" (default): fit of a Gaussian on top of a linear background. The most accurate and robust option.
            - "moments": first moment (intensity-weighted mean position) of the background-subtracted profile within +-trace_half_width.
            - "argmax": position of the brightest pixel, refined to sub-pixel precision with a parabola.
            - "multi": same as "profile", but for all the spectra detected in the image (not only the brightest one).
        y_pixel_range: integer
            Half-size (in pixels, along the y axis) of the window in which the center of the trace is searched in each bin. It must be larger
            than the half-width of the spatial profile and, for method = "multi", it is automatically reduced to half the distance between
            neighbouring spectra.
        xlims: list or array
            Optional. The two limits in the x axis of the region where the trace is measured, e.g. xlims = [700, 1800] if your spectrum has
            signal only between these pixels. The fitted trace is defined (extrapolated) over the whole x axis.
        poly_order: integer
            The order of the polynomial(s) used to fit the trace.
        trace_half_width: integer
            The half-width of the trace. Used by the "moments" method, to separate the trace from the local background, and in the plots.
        peak_height: float
            Optional. Minimum height (in counts above the background) of the spatial profile of a spectrum for it to be detected. If None
            (default), the threshold is 'detection_sigma' times the noise of the spatial profile.
        distance: float
            Minimum distance (in pixels, along the y axis) between neighbouring spectra. Default is 50 pixels.
        Number_of_slices, peak_dispersion_limit:
            Deprecated and ignored (kept for compatibility with previous versions). The x range used for each trace is now selected
            automatically from the significance of the detection in each bin (see 'min_snr').
        repeat_trace_at: dict
            This variable allows the user to duplicate a specific trace and place the copy at an arbitrary position along the y-axis. For example,
            if two spectra are detected, you can replicate one or both using a dictionary such as repeat_trace_at = {"spec_0": 75, "spec_1": -50},
            where the numerical values represent the vertical shifts (in pixels) relative to the original spec_0 and/or spec_1 traces. This feature
            is particularly useful when analyzing spectra without a clear continuum, such as those from H II regions.
        main_plot: boolean
            If True, easyspec will plot all the traces over the image.
        plot_residuals: boolean
            If True, easyspec will plot, for each trace, a zoom of the image around it and the residuals of the polynomial fit.
        bin_size: integer
            Optional. Number of columns combined in each bin. Default: ~1% of the number of columns (at least 5). Use larger values for faint
            spectra.
        min_snr: float
            Minimum signal-to-noise ratio of the detection in a bin for its center to be used in the fit. Default is 3.
        detection_sigma: float
            Detection threshold (in units of the noise of the spatial profile) used when peak_height is None. Default is 5.
        clip_sigma: float
            Threshold (in units of the scatter of the residuals) of the iterative sigma clipping of the polynomial fit. Default is 3.
        max_iterations: integer
            Maximum number of iterations of the sigma clipping. Default is 10.

        Returns
        -------
        fitted_polymodel_list: list
            List with the fitted trace (astropy Polynomial1D model, callable as trace(x)) of each spectrum, sorted from the bottom to the top of
            the image.

        Notes
        -----
        The measurements and diagnostics of each trace (bin centers, measured centers and their errors, Gaussian width of the spatial
        profile, bins used in the fit, RMS of the residuals) are stored in the attribute 'self.trace_info' (a list of dictionaries, in the same
        order as fitted_polymodel_list).
        """

        # ---------------------------------------------------------------- Input checks
        if self.aspect_ratio is None:
            raise NameError("The image dimensions are unknown. Please load your data with the function extraction.import_data() first.")
        image = np.asarray(target_spec_data, dtype=float)
        if image.ndim != 2:
            raise ValueError(f"'target_spec_data' must be a 2D image, but it has shape {image.shape}.")
        ny, nx = image.shape
        valid_methods = ("profile", "moments", "argmax", "multi")
        if method not in valid_methods:
            raise ValueError(f"Wrong method name '{method}'. Accepted entries: " + ", ".join(f"'{m}'" for m in valid_methods) + ".")
        if not (isinstance(poly_order, (int, np.integer)) and poly_order >= 0):
            raise ValueError("'poly_order' must be a non-negative integer.")
        if not (y_pixel_range >= 3):
            raise ValueError("'y_pixel_range' must be at least 3 pixels.")
        if Number_of_slices is not None or peak_dispersion_limit is not None:
            warnings.warn("The parameters 'Number_of_slices' and 'peak_dispersion_limit' are deprecated and ignored. The x range of each trace "
                          "is now selected automatically from the significance of the detection in each bin (see 'min_snr').", DeprecationWarning)
        if xlims is None:
            x_min, x_max = 0, nx - 1
        else:
            if len(xlims) != 2:
                raise ValueError("'xlims' must be a list with two values, e.g. xlims = [100, 1900].")
            x_min, x_max = int(max(0, np.floor(min(xlims)))), int(min(nx - 1, np.ceil(max(xlims))))
            if x_max - x_min < 10:
                raise ValueError(f"The region defined by xlims = {list(xlims)} is too small (or outside the image, which has {nx} columns).")
        if bin_size is None:
            bin_size = max(5, int(round(nx/100)))
        bin_size = int(max(1, min(bin_size, (x_max - x_min + 1)//3)))

        variance, mask = self._image_quality_arrays(target_spec_data)
        good = np.isfinite(image)
        if mask is not None:
            good &= (mask & (1 | 4 | 8)) == 0   # Cosmic rays, non-finite values and pixels without data after the alignment
        if variance is not None:
            variance = np.asarray(variance, dtype=float)
            good &= np.isfinite(variance) & (variance > 0)
        measure_method = "profile" if method == "multi" else method

        # ---------------------------------------------------------------- 1. Detection
        y_starts, x_start = self._detect_traces(image, good, method, x_min, x_max, peak_height, distance, detection_sigma, y_pixel_range)

        # ---------------------------------------------------------------- 2. Centers in bins of columns
        x_bins, binned, binned_variance = self._bin_columns(image, good, variance, bin_size, x_min, x_max)
        fitted_polymodel_list, trace_info = [], []
        for n, y0 in enumerate(y_starts):
            half_window = float(y_pixel_range)
            neighbours = np.delete(y_starts, n)
            if len(neighbours) > 0:   # The window never reaches half the way to a neighbouring spectrum
                half_window = min(half_window, 0.5*np.min(np.abs(neighbours - y0)))
            if half_window < 3:
                warnings.warn(f"Spectrum {n} is closer than 6 pixels to another spectrum, so its trace may be unreliable.")
                half_window = 3.0
            centers, errors, sigmas, snrs = self._follow_trace(x_bins, binned, binned_variance, y0, x_start, half_window, trace_half_width,
                                                               measure_method, min_snr)

            # ---------------------------------------------------------- 3. Polynomial fit
            try:
                model, used, rms = self._fit_trace(x_bins, centers, errors, poly_order, clip_sigma, max_iterations, nx)
            except RuntimeError as error:
                if method == "multi" and len(y_starts) > 1:
                    warnings.warn(f"The candidate spectrum at y = {y0:.1f} was discarded: {error}")
                    continue
                raise
            fitted_polymodel_list.append(model)
            trace_info.append({"x_bins": x_bins, "centers": centers, "center_errors": errors, "profile_sigma": sigmas, "snr": snrs,
                               "used_in_fit": used, "rms": rms, "initial_y": float(y0), "half_window": half_window,
                               "x_range_fitted": (float(x_bins[used].min()), float(x_bins[used].max())), "method": measure_method,
                               "bin_size": bin_size})

        # Remove duplicated traces (two candidate positions that converged to the same spectrum), keeping the one with more bins in the fit:
        keep = list(range(len(fitted_polymodel_list)))
        for i in range(len(fitted_polymodel_list)):
            for j in range(i + 1, len(fitted_polymodel_list)):
                if i not in keep or j not in keep:
                    continue
                common = trace_info[i]["used_in_fit"] & trace_info[j]["used_in_fit"]
                x_common = x_bins[common] if np.any(common) else x_bins
                separation = np.median(np.abs(fitted_polymodel_list[i](x_common) - fitted_polymodel_list[j](x_common)))
                if separation < max(3.0, 0.5*min(trace_info[i]["half_window"], trace_info[j]["half_window"])):
                    drop = j if np.count_nonzero(trace_info[i]["used_in_fit"]) >= np.count_nonzero(trace_info[j]["used_in_fit"]) else i
                    keep.remove(drop)
        fitted_polymodel_list = [fitted_polymodel_list[i] for i in keep]
        trace_info = [trace_info[i] for i in keep]

        if len(fitted_polymodel_list) == 0:
            raise RuntimeError("No trace could be fitted. Try a larger 'bin_size', a lower 'min_snr' or 'peak_height', or check 'xlims'.")

        # ---------------------------------------------------------------- Repeated traces
        if repeat_trace_at is not None:
            fitted_polymodel_list, trace_info = self._repeat_trace(fitted_polymodel_list, trace_info, repeat_trace_at)
        else:
            order = np.argsort([model((x_min + x_max)/2) for model in fitted_polymodel_list])
            fitted_polymodel_list = [fitted_polymodel_list[i] for i in order]
            trace_info = [trace_info[i] for i in order]
        self.trace_info = trace_info

        # ---------------------------------------------------------------- Summary and sanity checks
        xvals = np.arange(nx)
        for n, (model, info) in enumerate(zip(fitted_polymodel_list, trace_info)):
            trace = model(xvals)
            sigma_profile = np.nanmedian(info["profile_sigma"][info["used_in_fit"]]) if np.any(np.isfinite(info["profile_sigma"])) else np.nan
            text = (f"Spectrum {n}: y = {trace[nx//2]:.2f} at x = {nx//2}; {np.count_nonzero(info['used_in_fit'])} of {len(x_bins)} bins used "
                    f"(x = {info['x_range_fitted'][0]:.0f}-{info['x_range_fitted'][1]:.0f}); residual RMS = {info['rms']:.3f} px")
            if np.isfinite(sigma_profile):
                text += f"; spatial FWHM = {2.3548*sigma_profile:.2f} px"
            if "repeated_from" in info:
                text += f" (copy of {info['repeated_from']} shifted by {info['shift']:+g} px)"
            print(text)
            if np.any(trace - trace_half_width < 0) or np.any(trace + trace_half_width > ny - 1):
                warnings.warn(f"The trace of spectrum {n} (+-trace_half_width) leaves the image in some columns (the polynomial may be extrapolated "
                              "far from the region where it was measured). Consider using 'xlims' or a lower 'poly_order'.")

        # ---------------------------------------------------------------- Plots
        if main_plot:
            plt.figure(figsize=(12,12*self.aspect_ratio))
            plt.imshow(_cleaning._log_display(image), cmap = "gray", origin='lower')
            plt.gca().set_aspect(1)
            if len(fitted_polymodel_list) > 1:
                plt.title(f"We have a total of {len(fitted_polymodel_list)} detected spectra")
            else:
                plt.title("We have detected 1 spectrum")
            for n, model in enumerate(fitted_polymodel_list):
                trace = model(xvals)
                plt.plot(xvals, trace, 'C0')
                plt.fill_between(xvals, trace-trace_half_width, trace+trace_half_width, color='C1', alpha=0.3)
                plt.text(nx*0.1, trace[int(nx*0.1)] + trace_half_width + 2, f"Spec {n}", color="limegreen", fontsize=14)

        if plot_residuals:
            for n, (model, info) in enumerate(zip(fitted_polymodel_list, trace_info)):
                trace = model(xvals)
                y1 = int(max(0, np.floor(np.min(trace)) - y_pixel_range))
                y2 = int(min(ny, np.ceil(np.max(trace)) + y_pixel_range + 1))
                plt.figure(figsize=(12,12*self.aspect_ratio))
                plt.imshow(_cleaning._log_display(image[y1:y2,:]), extent=[-0.5, nx-0.5, y1-0.5, y2-0.5], cmap = "gray", origin='lower', aspect="auto")
                plt.plot(xvals, trace, 'C0')
                plt.fill_between(xvals, trace-trace_half_width, trace+trace_half_width, color='C1', alpha=0.15)
                plt.title(f"Spectrum {n} - "+str(self.target_name)+" field")

                plt.figure(figsize=(12,5))
                used = info["used_in_fit"]
                measured = np.isfinite(info["centers"])
                residuals = info["centers"] - model(info["x_bins"])
                plt.errorbar(info["x_bins"][used], residuals[used], yerr=info["center_errors"][used], fmt='o', ms=3, label="Used in the fit")
                rejected = measured & ~used
                if np.any(rejected):
                    plt.plot(info["x_bins"][rejected], residuals[rejected], 'x', color="red", label="Rejected (sigma clipping)")
                plt.axhline(0, color="k", linestyle="--")
                plt.ylabel("Trace residuals (data-model) [px]")
                plt.xlabel("x (pixels)")
                plt.title(f"Spectrum {n} - Trace residuals (RMS = {info['rms']:.3f} px)")
                plt.grid(linestyle=":")
                plt.legend()

        if main_plot or plot_residuals:
            plt.show()

        return fitted_polymodel_list
    
    
    @staticmethod
    def _pairs_per_trace(value, n_traces, name):

        """
        Convert a parameter given as a number, a list [above, below], or a list of N pairs [[above, below], ...] into a list with one
        (above, below) pair per trace. Used for shift_y_pixels and background_gap in extracting().
        """

        if isinstance(value, (int, float, np.integer, np.floating)):
            return [(float(value), float(value))]*n_traces
        if isinstance(value, (list, tuple, np.ndarray)):
            shape = np.shape(value)
            if shape == (2,):
                return [(float(value[0]), float(value[1]))]*n_traces
            if shape == (n_traces, 2):
                return [(float(a), float(b)) for a, b in value]
        raise TypeError(f"Invalid input for {name}: {value!r}. This parameter must be a number, a list with 2 numbers [above, below], or a list "
                        f"with N pairs of numbers given as sublists (e.g. [[30, 70], [40, 20]]), where N = {n_traces} is the number of traces "
                        "in fitted_polymodel_list.")

    @_plot_style
    def extracting(self, target_spec_data, fitted_polymodel_list, data_type, MC_steps = None, master_lamp_data = None, trace_half_width = 10, shift_y_pixels = 30,
                   extraction_weights = "optimal", lamp_peak_height = None, peak_distance = None, diagnostic_plots = True, spec_plots = True,
                   background_gap = 3, background_order = 1, profile_x_order = None, reject_sigma = 5.0, refine_lamp_peaks = True):

        """
        This function extracts the 1D spectrum along each trace, with the optimal extraction algorithm of Horne (1986, PASP 98, 609), and
        estimates the uncertainty (variance) of each spectral pixel.

        For each trace:

        1. Sky: in each column, a polynomial (of order 'background_order') of the distance to the trace is fitted to the sky pixels on both
           sides of the trace, from trace_half_width + background_gap up to shift_y_pixels pixels from the trace (both may differ above and
           below the trace, and from one trace to another), with sigma clipping. The sky
           regions follow the trace, and pixels near other traces are excluded from them. The uncertainty of the sky fit is propagated to the
           spectrum.
        2. Spatial profile: the fraction of the flux falling in each pixel is measured from the data themselves (no Gaussian assumption), as a
           smooth function of the distance to the trace, which may vary slowly along the dispersion axis.
        3. Extraction: each pixel of the aperture (+-trace_half_width around the trace, with fractional pixels at its edges) is weighted by
           P/V (profile/variance), which maximizes the signal-to-noise ratio of the spectrum without biasing the flux. Cosmic rays and bad pixels
           (from the data quality mask, and outliers with respect to the profile) are rejected, and the variance is computed from the model of
           the spectrum, which avoids the bias that noisy variances would cause.

        With extraction_weights = "tophat", the pixels of the same aperture are simply summed (masked pixels are corrected with the spatial
        profile). Both options estimate the same quantity (the flux within the aperture), so the target and the standard star may be compared
        directly, but the same option and the same trace_half_width should be used for both.

        If the image was loaded with import_data() from a file produced by easyspec.cleaning(), its variance ("VAR") and data quality mask
        ("MASK") are used. Otherwise, the noise is estimated from the scatter of the sky pixels plus the Poisson noise of the spectrum (assuming
        the image is in electrons).

        In case you need references for the lamp peaks, here are some databases you can use (as of 12 September, 2024):
         * TNG DOLORES spectrograph reference lamps can be found in table 1 here: https://www.tng.iac.es/instruments/lrs/
         * SOAR Goodman Spectrograph Reference Lamp Library: https://soardocs.readthedocs.io/projects/lamps/en/latest/
         * Observatorio do Pico dos Dias (OPD-LNA) spectrographs: https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd/instrumentacao/instrumentos-e-detectores

        Parameters
        ----------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target (or standard star) spectral image.
        fitted_polymodel_list: list
            List with the fitted traces for one or more spectra in the image. This is the output of the function extraction.tracing().
        data_type: string
            Options are "target" or "std_star".
        MC_steps: int
            Deprecated and ignored. The uncertainty of the spectrum is now computed analytically (see Returns and Notes).
        master_lamp_data: string
            Optional. The path to the master lamp data file. This file can be produced with the easyspec cleaning() class.
        trace_half_width: float
            Half-width (in pixels) of the extraction aperture, i.e. the spectra are extracted within +-trace_half_width pixels from their traces.
            For the optimal extraction, a generous value (e.g. ~2-3 times the FWHM of the spatial profile) is recommended: the faint wings of the
            profile get small weights, so a wide aperture does not add noise.
        shift_y_pixels: integer, list with 2 integers, or list with N pairs of integers
            Outer limit (in pixels from the trace) of the sky regions. An integer applies to both sides; a list [above, below] gives different
            limits above and below the trace; a list of N pairs gives the limits for each of the N traces.
        extraction_weights: string
            "optimal" (default) for the optimal extraction, or "tophat" for a simple sum over the aperture. "gaussian" (from previous versions)
            is accepted as an alias of "optimal".
        lamp_peak_height: float
            The height of the lamp peaks (in counts and with respect to zero) to be detected. Default: the mean of the lamp spectrum.
        peak_distance: float
            The minimum distance between lamp peaks in pixels. Default is the length of the x-axis divided by 20.
        diagnostic_plots: boolean
            If True, easyspec will plot the image with the extraction aperture and sky regions, and the measured spatial profile.
        spec_plots: boolean
            If True, easyspec will plot the extracted spectra (with their 1-sigma uncertainties) and the lamp spectra.
        background_gap: float, list with 2 floats, or list with N pairs of floats
            Gap (in pixels) between the edge of the extraction aperture and the start of the sky regions, i.e. the sky regions start at
            trace_half_width + background_gap pixels from the trace. It accepts the same formats as shift_y_pixels: a number applies to both
            sides of all traces; a list [above, below] gives different gaps above and below the trace; a list of N pairs, e.g.
            [[3, 10], [5, 5]], gives the gaps for each of the N traces. Default is 3.
        background_order: int
            Order of the polynomial fitted to the sky along the slit in each column (0 = constant, 1 = linear gradient). Default is 1.
        profile_x_order: int
            Optional. Order of the polynomial that describes the variation of the spatial profile along the dispersion axis (0 = the same
            profile in all columns). Default: chosen automatically from the total signal-to-noise ratio of the spectrum (0 for S/N < 100, 1
            below 300, 2 below 1000, and 3 above).
        reject_sigma: float
            Threshold (in units of the noise) for the rejection of outlier pixels (cosmic rays, hot pixels) in the optimal extraction. Default is 5.
        refine_lamp_peaks: boolean
            If True (default), the positions of the lamp peaks are refined to sub-pixel precision with a Gaussian fit, which improves the
            wavelength solution. The number and order of the peaks are not changed.

        Returns
        -------
        spec_list: list
            List with the extracted spectra (one array per trace, in counts/electrons per pixel of the dispersion axis).
        lamp_spec_list: list
            Optional (only if master_lamp_data is given). List with the extracted lamp spectra, along the same apertures.
        lamp_peak_positions_list: list
            Optional (only if master_lamp_data is given). List with the positions (in pixels) of the peaks of each lamp spectrum.

        Notes
        -----
        The variance and the data quality mask of each spectrum are stored in the attributes self.spec_variance_list and
        self.spec_mask_list (for data_type = "target") or self.std_spec_variance_list and self.std_spec_mask_list (for "std_star"). The bits
        of the 1D mask are described in optimal_extraction.SPECTRUM_MASK_BITS. Diagnostics of each extraction (sky regions, spatial profile,
        signal-to-noise ratio, rejected pixels, etc.) are stored in self.extraction_info (or self.std_extraction_info).

        For compatibility with the calibration functions of previous versions, the 1-sigma uncertainty of the target spectra is also stored
        in self.spec_systematic_error_list, which is propagated by extinction_correction() and target_flux_calibration() and saved in the
        output files.
        """

        # ---------------------------------------------------------------- Input checks
        if self.aspect_ratio is None:
            raise NameError("The image dimensions are unknown. Please load your data with the function extraction.import_data() first.")
        if data_type not in ("target", "std_star"):
            raise ValueError(f"Invalid data_type '{data_type}'. Options are 'target' or 'std_star'.")
        if MC_steps is not None:
            warnings.warn("The parameter 'MC_steps' is deprecated and ignored: the uncertainty of the spectrum is now computed analytically.",
                          DeprecationWarning)
        if extraction_weights == "gaussian":
            print("Note: extraction_weights='gaussian' is now an alias of 'optimal' (optimal extraction with an empirical spatial profile).")
            extraction_weights = "optimal"
        if extraction_weights not in ("optimal", "tophat"):
            raise ValueError(f"The variable 'extraction_weights' must be 'optimal' or 'tophat'. Instead, the input string was '{extraction_weights}'.")
        if data_type == "std_star" and self.extraction_weights is not None and extraction_weights != self.extraction_weights:
            warnings.warn(f"The target was extracted with extraction_weights = '{self.extraction_weights}', but the standard star is being "
                          f"extracted with '{extraction_weights}'. Please use the same option (and the same trace_half_width) for both.")
        if not trace_half_width > 0:
            raise ValueError("'trace_half_width' must be positive.")
        if not (isinstance(background_order, (int, np.integer)) and 0 <= background_order <= 3):
            raise ValueError("'background_order' must be an integer between 0 and 3.")

        image = np.asarray(target_spec_data, dtype=float)
        ny, nx = image.shape
        n_traces = len(fitted_polymodel_list)
        if n_traces == 0:
            raise ValueError("'fitted_polymodel_list' is empty.")

        # Sky regions [above, below] for each trace: they go from trace_half_width + background_gap (inner limit) to shift_y_pixels (outer limit)
        sky_limits = self._pairs_per_trace(shift_y_pixels, n_traces, "shift_y_pixels")
        gaps = self._pairs_per_trace(background_gap, n_traces, "background_gap")
        inner_limits = []
        for n, ((above, below), (gap_above, gap_below)) in enumerate(zip(sky_limits, gaps)):
            if min(gap_above, gap_below) < 0:
                raise ValueError(f"background_gap must be >= 0 (spectrum {n}: [above, below] = [{gap_above:g}, {gap_below:g}]).")
            inner_above, inner_below = trace_half_width + gap_above, trace_half_width + gap_below
            for side, inner, outer in (("above", inner_above, above), ("below", inner_below, below)):
                if outer < inner + 3:
                    raise ValueError(f"The sky region {side} spectrum {n} is too narrow: shift_y_pixels must be at least trace_half_width + "
                                     f"background_gap + 3 = {inner + 3:g} pixels on this side (it is {outer:g}). Increase shift_y_pixels or "
                                     "decrease background_gap.")
            inner_limits.append((inner_above, inner_below))

        variance_image, mask_image = self._image_quality_arrays(target_spec_data)
        has_variance = variance_image is not None
        if mask_image is None:
            mask_image = np.zeros((ny, nx), dtype=np.uint8)
        if not has_variance:
            print("Note: the image has no variance ('VAR' extension). The noise will be estimated from the scatter of the sky pixels plus the "
                  "Poisson noise of the spectrum, assuming the image is in electrons.")

        lamp_image = None
        if master_lamp_data is not None:
            lamp_image, _, _, _ = data_io.read_reduced_frame(master_lamp_data, estimate_variance=False)
            if lamp_image.shape != image.shape:
                raise ValueError(f"The master lamp has shape {lamp_image.shape}, but the image has shape {image.shape}. Both must be reduced with "
                                 "the same trim (and, if the images were aligned, the master lamp must be built from the aligned data).")

        xvals = np.arange(nx)
        traces = np.array([model(xvals) for model in fitted_polymodel_list])
        spec_list, variance_list, spec_mask_list, info_list = [], [], [], []
        lamp_spec_list, lamp_peak_positions_list, lamp_peak_heights_list = [], [], []

        for number, trace in enumerate(traces):
            above, below = sky_limits[number]
            inner_above, inner_below = inner_limits[number]
            outer = max(above, below)
            y1 = int(max(0, np.floor(np.min(trace) - outer) - 1))
            y2 = int(min(ny, np.ceil(np.max(trace) + outer) + 2))
            rows = np.arange(y1, y2)
            data = image[y1:y2]
            variance = np.asarray(variance_image[y1:y2], dtype=float) if has_variance else None
            bits = mask_image[y1:y2]
            good = np.isfinite(data) & ((bits & optimal_extraction.BAD_PIXEL_BITS) == 0)
            if variance is not None:
                good &= np.isfinite(variance) & (variance > 0)
            dy = rows[:, None] - trace[None, :]

            # Columns where the aperture or the sky region falls outside the image:
            outside = (trace + above > ny - 1) | (trace - below < 0)

            # ---------------------------------------------------- 1. Sky
            sky_region = ((dy >= inner_above) & (dy <= above)) | ((dy <= -inner_below) & (dy >= -below))
            for m, other in enumerate(traces):   # Exclude the pixels near other traces (their aperture plus gap) from the sky regions
                if m != number:
                    sky_region &= np.abs(rows[:, None] - other[None, :]) > max(inner_limits[m])
            # ---------------------------------------------------- Variance model (see optimal_extraction.estimate_variance_model)
            gain_factor = optimal_extraction.estimate_variance_model(data, variance, good) if variance is not None else None
            sky, sky_variance, sky_covariance, sky_ok, sky_used = optimal_extraction.fit_sky(data, variance, good, dy, sky_region,
                                                                                              order=background_order, gain_factor=gain_factor)
            sky_design = optimal_extraction._polynomial_design(dy, background_order)
            if variance is None:
                # Noise of each column from the scatter of its sky pixels, plus the Poisson noise of the signal (image assumed in electrons).
                # The Poisson part is computed from the data, so it is later replaced by the model (gain_factor = 1).
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sky_scatter = 1.4826*np.nanmedian(np.abs(np.where(sky_used, data - sky, np.nan)), axis=0)
                sky_scatter = np.where(np.isfinite(sky_scatter) & (sky_scatter > 0), sky_scatter, np.nanmedian(sky_scatter))
                variance = sky_scatter[None, :]**2 + np.clip(data - np.nan_to_num(sky), 0, None)
                gain_factor = 1.0
            # Variance of each pixel from the model of the sky (it is refined with the model of the spectrum in the extraction):
            variance_model = optimal_extraction.model_variance(variance, data, np.nan_to_num(sky), gain_factor)

            # ---------------------------------------------------- 2. Spatial profile
            coverage = optimal_extraction.aperture_coverage(dy, trace_half_width)
            in_aperture = coverage > 0
            signal = data - sky
            column_complete = np.all(~in_aperture | good, axis=0) & sky_ok
            with np.errstate(invalid="ignore"):
                box_flux = np.sum(np.where(in_aperture, coverage*signal, 0.0), axis=0)
                box_sigma = np.sqrt(np.sum(np.where(in_aperture, coverage**2*(variance_model + sky_variance), 0.0), axis=0))
            box_snr = np.where(column_complete & (box_sigma > 0), box_flux/box_sigma, 0.0)
            profile_columns = column_complete & (box_snr > 2)
            total_snr2 = np.sum(box_snr[profile_columns]**2)
            # Order of the variation of the profile along x: higher for brighter spectra (from 0 = constant up to 3)
            total_snr = np.sqrt(total_snr2)
            x_order = profile_x_order if profile_x_order is not None else int(np.searchsorted([100, 300, 1000], total_snr))
            profile_model = None
            if np.count_nonzero(profile_columns) >= 10:
                with np.errstate(invalid="ignore", divide="ignore"):
                    values = np.where(profile_columns[None, :] & in_aperture & good, signal/box_flux[None, :], np.nan)
                    # Each pixel is weighted by the S/N of its column (the same weight for all pixels of a column), which avoids the bias
                    # that per-pixel weights computed from the noisy data would introduce in the profile:
                    weights = np.where(np.isfinite(values), box_snr[None, :], 0.0)
                try:
                    profile_model = optimal_extraction.EmpiricalProfile(dy, values, weights, xvals, x_order, trace_half_width, nx)
                except RuntimeError:
                    profile_model = None
            if profile_model is not None:
                profile = profile_model.evaluate(dy, xvals)
                profile_source = f"empirical (FWHM = {profile_model.fwhm:.1f} px, order {x_order} along x)"
            else:
                # Faint spectrum: Gaussian profile with the width measured by tracing() (if available)
                sigma = np.nan
                info = getattr(self, "trace_info", None)
                if info is not None and len(info) == n_traces:
                    sigma = np.nanmedian(info[number]["profile_sigma"])
                if not np.isfinite(sigma):
                    sigma = trace_half_width/3
                profile = np.exp(-0.5*(dy/sigma)**2)*in_aperture
                profile_source = f"Gaussian (sigma = {sigma:.2f} px; the spectrum is too faint to measure its profile)"
                warnings.warn(f"Spectrum {number}: the signal is too faint to measure the spatial profile, so a Gaussian profile was used.")
            normalization = np.sum(coverage*profile, axis=0)
            profile = np.where(normalization[None, :] > 0, profile/np.where(normalization > 0, normalization, 1.0)[None, :], 0.0)

            # ---------------------------------------------------- 3. Extraction
            flux_opt, variance_opt, used, final_pixel_variance = optimal_extraction.horne_extraction(data, sky, variance, profile, coverage, good,
                                                                                                      sky_design, sky_covariance,
                                                                                                      reject_sigma=reject_sigma,
                                                                                                      gain_factor=gain_factor)
            flux_box, variance_box = optimal_extraction.boxcar_extraction(data, sky, final_pixel_variance, profile, coverage, used, sky_design,
                                                                          sky_covariance)
            if extraction_weights == "optimal":
                flux, flux_variance = flux_opt, variance_opt
            else:
                flux, flux_variance = flux_box, variance_box

            # ---------------------------------------------------- 1D data quality mask
            lost_fraction = 1.0 - np.sum(np.where(used, coverage*profile, 0.0), axis=0)
            rejected = in_aperture & good & ~used
            spec_mask = np.zeros(nx, dtype=np.uint8)
            spec_mask[lost_fraction > 0.3] |= 1
            spec_mask[np.any(in_aperture & ((bits & 2) > 0), axis=0)] |= 2
            invalid = ~np.isfinite(flux) | ~np.isfinite(flux_variance) | (flux_variance <= 0) | ~sky_ok
            spec_mask[invalid] |= 4
            spec_mask[outside] |= 8
            flux = np.where(invalid, np.nan, flux)
            flux_variance = np.where(invalid, np.nan, flux_variance)

            spec_list.append(flux)
            variance_list.append(flux_variance)
            spec_mask_list.append(spec_mask)
            snr = np.where(np.isfinite(flux) & (flux_variance > 0), flux/np.sqrt(flux_variance), np.nan)
            info_list.append({"trace": trace, "aperture_half_width": trace_half_width, "sky_limits": {"inner": (inner_above, inner_below), "outer": (above, below)},
                              "background_order": background_order, "profile": profile_source, "profile_model": profile_model,
                              "median_snr": float(np.nanmedian(snr)), "n_rejected_pixels": int(np.count_nonzero(rejected)),
                              "flux_tophat": flux_box, "variance_tophat": variance_box, "flux_optimal": flux_opt, "variance_optimal": variance_opt,
                              "variance_model": f"VAR + b*(model - data), b = {gain_factor:.3g}" if gain_factor else "VAR as given",
                              "gain_factor": gain_factor})
            with np.errstate(invalid="ignore", divide="ignore"):
                gain = np.nanmedian((flux_variance/variance_box)[np.isfinite(flux_box) & (flux_box/np.sqrt(variance_box) > 3)]) if extraction_weights == "optimal" else np.nan
            print(f"Spectrum {number}: median S/N per pixel = {np.nanmedian(snr):.1f}; profile: {profile_source}; "
                  f"{np.count_nonzero(rejected)} outlier pixel(s) rejected"
                  + (f"; variance relative to a simple sum = {gain:.2f}" if np.isfinite(gain) else "") + ".")

            # ---------------------------------------------------- Lamp
            if lamp_image is not None:
                lamp_band = lamp_image[y1:y2]
                lamp_spectrum = np.sum(np.where(in_aperture & np.isfinite(lamp_band), coverage*lamp_band, 0.0), axis=0)
                height = lamp_peak_height if lamp_peak_height is not None else np.mean(lamp_spectrum)
                distance = peak_distance if peak_distance is not None else int(nx/20)
                lamp_peak_positions, lamp_peak_heights = scipy.signal.find_peaks(lamp_spectrum, distance=distance, height=height)
                if refine_lamp_peaks:
                    lamp_peak_positions = optimal_extraction.refine_peak_positions(lamp_spectrum, lamp_peak_positions)
                lamp_spec_list.append(lamp_spectrum)
                lamp_peak_positions_list.append(lamp_peak_positions)
                lamp_peak_heights_list.append(lamp_peak_heights)

            # ---------------------------------------------------- Plots
            if spec_plots:
                sigma_flux = np.sqrt(flux_variance)
                plt.figure(figsize=(12,5))
                plt.fill_between(xvals, flux - sigma_flux, flux + sigma_flux, color='C0', alpha=0.3, linewidth=0, label=r"$\pm1\sigma$")
                plt.plot(xvals, flux, label=f'{extraction_weights} extraction, spec {number}', alpha=0.9, linewidth=0.6, color='orange')
                flagged = (spec_mask & (1 | 4)) > 0
                if np.any(flagged):
                    plt.plot(xvals[flagged], np.nan_to_num(flux[flagged]), 'x', color="red", ms=3, label="Flagged pixels")
                plt.title(f"Non-calibrated {extraction_weights}-extracted spectrum {number} - "+str(self.target_name)+" field")
                plt.grid(linestyle=":",which="both")
                plt.minorticks_on()
                plt.ylabel("Counts")
                plt.xlabel("Pixels")
                plt.legend(loc='upper left')

                if lamp_image is not None:
                    plt.figure(figsize=(12,5))
                    plt.plot(lamp_spectrum, label='Lamp spec', alpha=0.9, linewidth=0.5, color='orange')
                    plt.title(f"Lamp spectrum - based on the trace of spec {number}")
                    plt.grid(linestyle=":",which="both")
                    plt.minorticks_on()
                    plt.ylabel("Counts")
                    plt.xlabel("Pixels")
                    plt.legend(loc='upper left')
                    for n in range(len(lamp_peak_positions)):
                        plt.text(lamp_peak_positions[n], lamp_peak_heights["peak_heights"][n], f"{lamp_peak_positions[n]:.2f}", rotation="vertical")

            if diagnostic_plots:
                plt.figure(figsize=(12,12*self.aspect_ratio))
                plt.imshow(_cleaning._log_display(image), cmap = "gray", origin='lower')
                plt.plot(xvals, trace, 'C0', linewidth=1, label=f"Trace spec {number}")
                plt.fill_between(xvals, trace-trace_half_width, trace+trace_half_width, color='C1', alpha=0.4, label="Extraction aperture")
                plt.fill_between(xvals, trace+inner_above, trace+above, color='magenta', alpha=0.3, label="Sky regions")
                plt.fill_between(xvals, trace-below, trace-inner_below, color='magenta', alpha=0.3)
                rejected_y, rejected_x = np.nonzero(rejected)
                if len(rejected_x) > 0:
                    plt.plot(rejected_x, rows[rejected_y], '.', color="red", ms=2, label="Rejected pixels")
                plt.title(f"Spectrum {number} - "+str(self.target_name)+" field")
                plt.legend()

                plt.figure(figsize=(12,5))
                show = in_aperture & good & profile_columns[None, :] if np.any(profile_columns) else in_aperture & good
                with np.errstate(invalid="ignore", divide="ignore"):
                    normalized = signal/box_flux[None, :]
                plt.plot(dy[show], normalized[show], ',', color="gray", alpha=0.5, label="Data (pixel value / flux of the column)")
                grid = np.linspace(-trace_half_width - 0.5, trace_half_width + 0.5, 400)
                if profile_model is not None:
                    for b, x_plot in enumerate(np.linspace(xvals[profile_columns].min() if np.any(profile_columns) else 0,
                                                           xvals[profile_columns].max() if np.any(profile_columns) else nx - 1, 4)):
                        model_values = profile_model.profile_at(grid, x_plot)
                        model_values = model_values/np.sum(profile_model.profile_at(np.arange(-np.floor(trace_half_width), np.floor(trace_half_width) + 1), x_plot))
                        plt.plot(grid, model_values, color=plt.cm.viridis(b/3), label=f"Profile at x = {x_plot:.0f}")
                plt.axvline(-trace_half_width, color="k", linestyle=":")
                plt.axvline(trace_half_width, color="k", linestyle=":")
                plt.xlabel("Distance to the trace (pixels)")
                plt.ylabel("Fraction of the flux per pixel")
                plt.title(f"Spatial profile - spectrum {number}")
                ylim = np.nanpercentile(normalized[show], [0.5, 99.5]) if np.any(show) else (0, 1)
                plt.ylim(min(-0.02, ylim[0]), ylim[1]*1.2)
                plt.grid(linestyle=":")
                plt.legend(fontsize=9)

        if lamp_image is not None and diagnostic_plots:
            plt.figure(figsize=(12,12*self.aspect_ratio))
            plt.imshow(_cleaning._log_display(lamp_image), cmap = "gray", origin='lower')
            for number, trace in enumerate(traces):
                plt.plot(xvals, trace, 'C0', linewidth=1, label=f"Trace spec {number}")
                plt.fill_between(xvals, trace-trace_half_width, trace+trace_half_width, color='C1', alpha=0.4)
            plt.title("Lamp spectrum")
            plt.legend()

        if spec_plots or diagnostic_plots:
            plt.show()

        # ---------------------------------------------------------------- Store the results
        if data_type == "target":
            self.extraction_weights = extraction_weights
            self.spec_variance_list = variance_list
            self.spec_mask_list = spec_mask_list
            self.extraction_info = info_list
            # 1-sigma uncertainty, propagated (as a multiplicative quantity) by extinction_correction() and target_flux_calibration():
            self.spec_systematic_error_list = [np.sqrt(v) for v in variance_list]
        else:
            self.std_spec_variance_list = variance_list
            self.std_spec_mask_list = spec_mask_list
            self.std_extraction_info = info_list

        if master_lamp_data is None:
            return spec_list
        else:
            return spec_list, lamp_spec_list, lamp_peak_positions_list

    @_plot_style
    def identify_lamp_lines(self, lamp_spec_list, lamps, wavelength_range, medium = "air", poly_order = 3, detection_sigma = 5.0,
                            range_tolerance = 0.1, tolerance = None, max_rms_pixels = 0.15, plots = True):

        """
        Automatically identify the lines of the lamp spectra, so that the wavelength calibration does not require identifying them by hand
        (e.g. by comparison with the lamp atlas of the instrument).

        You only need to know which lamp(s) were used and, approximately (within ~10%), the wavelength range covered by the spectrum (e.g.
        from the setup of the instrument: grating and central wavelength). The lines are detected, matched to a catalogue of laboratory
        wavelengths by pattern matching, and refined with a polynomial solution (see the module line_identification for the details). The
        outputs can be passed directly to wavelength_calibration():

            positions, wavelengths = extraction.identify_lamp_lines(lamp_spec_list, lamps="HgArNe", wavelength_range=[3050, 7000])
            wavelengths_list, wavelengths_fit_rms_list = extraction.wavelength_calibration(positions, wavelengths, data_type="target")

        Always check the plots: the identified lines are labelled on the lamp spectrum. If the identification fails, a clear error is raised
        (instead of returning a wrong solution); then check the lamps and the wavelength range, or identify a few lines by hand.

        Parameters
        ----------
        lamp_spec_list: list
            List with the lamp spectra (output of extracting()).
        lamps: string, list, path or array
            The lamp(s): e.g. "HgArNe", "CuHeAr", "Ne", or ["HgI", "NeI"] (built-in line lists: HgI, NeI, ArI, HeI, CuI, CdI, ZnI, KrI, XeI,
            FeI, ThAr, from PypeIt). Give only the elements present in your lamp. Alternatively, the path to a text file with your own list of
            wavelengths (first column, in Angstroms and in the same medium as 'medium') or an array with these wavelengths.
        wavelength_range: list
            Approximate wavelengths (in Angstroms) at the first and at the last pixel of the spectrum, [lambda(x = 0), lambda(x = last)]. If the
            wavelength decreases along the x axis, give them in decreasing order.
        medium: string
            "air" (default; the usual convention for ground-based optical spectra) or "vacuum".
        poly_order: integer
            Order of the polynomial used during the identification (the final solution is fitted later by wavelength_calibration()).
        detection_sigma: float
            Detection threshold of the lamp lines, in units of the noise of the lamp spectrum. Default is 5.
        range_tolerance: float
            Relative uncertainty of 'wavelength_range'. Default is 0.1 (10% of the range).
        tolerance: float
            Optional. Final matching tolerance, in pixels. Default: the FWHM of the lamp lines (at least 1.5 pixels).
        max_rms_pixels: float
            A solution is accepted only if the RMS of the residuals of the identified lines is smaller than this value (in pixels) and than
            10% of the FWHM of the lines. Default is 0.15 pixels; increase it for low-resolution spectra with many blended lines.
        plots: boolean
            If True, easyspec plots each lamp spectrum with the identified lines and the residuals of the identification.

        Returns
        -------
        lamp_peak_positions_list: list
            For each lamp spectrum, the positions (in pixels) of all the detected peaks.
        corresponding_wavelengths_list: list
            For each lamp spectrum, the laboratory wavelength of each peak (np.nan for the peaks that were not identified).

        Notes
        -----
        The line lists were taken from PypeIt (https://github.com/pypeit/PypeIt; Prochaska et al. 2020, JOSS 5, 2308), which compiled them
        from NIST and instrument calibrations. Please cite PypeIt and NIST if you use them. The details of each identification are stored in
        self.lamp_identification_info.
        """

        print("The built-in line lists adopted here (HgI, NeI, ArI, HeI, CuI, CdI, ZnI, KrI, XeI, FeI, ThAr) were taken from PypeIt, which compiled them from NIST and instrument calibrations. Please cite PypeIt and NIST if you use them.")

        line_wavelengths, line_amplitudes, line_names = line_identification.load_line_list(lamps, medium=medium)
        if len(line_wavelengths) == 0:
            raise ValueError("The line list is empty.")
        if np.ndim(wavelength_range) != 1 or len(wavelength_range) != 2:
            raise ValueError("'wavelength_range' must be [lambda(first pixel), lambda(last pixel)], e.g. [3050, 7000].")

        positions_list, wavelengths_list, info_list = [], [], []
        previous_solution = None
        for number, lamp_spectrum in enumerate(lamp_spec_list):
            lamp_spectrum = np.asarray(lamp_spectrum, dtype=float)
            result = None
            if previous_solution is not None:
                # The other traces are identified starting from the solution of the first one (faster and consistent); if this fails,
                # the full search is done.
                try:
                    result = line_identification.identify_lines(lamp_spectrum, line_wavelengths, line_amplitudes, poly_order=poly_order,
                                                                detection_sigma=detection_sigma, tolerance=tolerance,
                                                                initial_solution=previous_solution, line_names=line_names,
                                                                max_rms_pixels=max_rms_pixels)
                except RuntimeError:
                    result = None
            if result is None:
                try:
                    result = line_identification.identify_lines(lamp_spectrum, line_wavelengths, line_amplitudes, wavelength_range=wavelength_range,
                                                                poly_order=poly_order, detection_sigma=detection_sigma,
                                                                range_tolerance=range_tolerance, tolerance=tolerance, line_names=line_names,
                                                                max_rms_pixels=max_rms_pixels)
                except RuntimeError as error:
                    raise RuntimeError(f"Lamp spectrum {number}: {error}")
            previous_solution = result["polynomial"]

            identified = np.isfinite(result["wavelengths"])
            names = np.array([line_names[np.argmin(np.abs(line_wavelengths - w))] if np.isfinite(w) else "" for w in result["wavelengths"]])
            x_identified = result["positions"][identified]
            coverage = np.ptp(x_identified)/(len(lamp_spectrum) - 1) if np.count_nonzero(identified) > 1 else 0.0
            solution = result["polynomial"]
            print(f"Lamp spectrum {number}: {len(result['positions'])} lines detected, {result['n_identified']} identified "
                  f"(RMS = {result['rms']:.3f} A = {result['rms_pixels']:.3f} px); lambda = {solution(0):.1f} to {solution(len(lamp_spectrum) - 1):.1f} A.")
            if coverage < 0.7:
                warnings.warn(f"Lamp spectrum {number}: the identified lines cover only {100*coverage:.0f}% of the detector (x = {x_identified.min():.0f}-"
                              f"{x_identified.max():.0f}), so the wavelength solution will be extrapolated beyond this range. Check whether your lamp "
                              "has lines there (e.g. add another element to 'lamps').")
            positions_list.append(result["positions"])
            wavelengths_list.append(result["wavelengths"])
            info_list.append(dict(result, names=names, coverage=coverage))

            if plots:
                x = np.arange(len(lamp_spectrum))
                figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
                top.plot(x, lamp_spectrum, color="C1", linewidth=0.6)
                peak_values = np.interp(result["positions"], x, lamp_spectrum)
                top.plot(result["positions"][~identified], peak_values[~identified], "v", color="gray", ms=5, label="Not identified")
                top.plot(result["positions"][identified], peak_values[identified], "v", color="C3", ms=6, label="Identified")
                for p, v, w, n in zip(result["positions"][identified], peak_values[identified], result["wavelengths"][identified], names[identified]):
                    top.text(p, v*1.05 + 0.01*np.nanmax(lamp_spectrum), f"{n} {w:.2f}", rotation=90, fontsize=8, ha="center", va="bottom")
                top.set_yscale("symlog", linthresh=max(1.0, np.nanmedian(np.abs(lamp_spectrum))))
                top.set_ylabel("Counts")
                top.set_title(f"Lamp spectrum {number} - identified lines ({medium} wavelengths, Angstrom)")
                top.legend(loc="upper right")
                top.grid(linestyle=":")
                top.set_ylim(0.5*np.min(lamp_spectrum),10*np.max(lamp_spectrum))
                residuals = result["wavelengths"] - solution(result["positions"])
                bottom.plot(result["positions"][identified], residuals[identified], "o", color="C3")
                bottom.axhline(0, color="k", linestyle="--")
                bottom.set_xlabel("x (pixels)")
                bottom.set_ylabel("Residuals [A]")
                bottom.grid(linestyle=":")

        # Consistency between traces: the solutions of different traces should differ only by a small shift/stretch (tilt of the lamp lines
        # on the detector). A larger difference usually means that a line near the edge was misidentified (or a blend was used) in one of them.
        if len(info_list) > 1:
            x = np.arange(len(lamp_spec_list[0]))
            reference = info_list[0]["polynomial"](x)
            for number, info in enumerate(info_list[1:], start=1):
                difference_pixels = (info["polynomial"](x) - reference)/np.median(np.abs(np.gradient(reference)))
                linear = np.polyval(np.polyfit(x, difference_pixels, 1), x)
                deviation = np.abs(difference_pixels - linear)
                if deviation.max() > 0.5:
                    warnings.warn(f"The wavelength solutions of lamp spectra 0 and {number} differ by more than a shift and a stretch (by up to "
                                  f"{deviation.max():.1f} px, near x = {x[np.argmax(deviation)]}). Check the identified lines near this position in "
                                  "the plots (e.g. blended lines), and the residuals of wavelength_calibration().")

        if plots:
            plt.show()
        self.lamp_identification_info = info_list
        return positions_list, wavelengths_list


    @_plot_style
    def wavelength_calibration(self, lamp_peak_positions_list, corresponding_wavelengths, data_type, poly_order = 2, diagnostic_plots = True,
                               clip_sigma = 4.0, max_iterations = 10, match_tolerance = 3.0):

        """
        This function fits the wavelength solution (wavelength as a function of the pixel position along the dispersion axis) of each
        spectrum, from the positions of the lamp lines and their known wavelengths.

        For each spectrum:

        1. The lamp lines are paired with the given wavelengths. If a spectrum has exactly one position per wavelength, they are paired in
           order, and the pairing is accepted if the resulting fit is good. Otherwise (e.g. if a line was not detected along one of the traces,
           or a spurious peak was detected), each wavelength is paired with the detected line closest to the position predicted by the best
           solution among the other spectra (corrected for a possible shift between the traces), within 'match_tolerance' pixels.
        2. A polynomial of order 'poly_order' is fitted to the pairs (in a normalized pixel coordinate, which keeps the fit well conditioned for
           any order), with iterative sigma clipping, so that misidentified or blended lines are rejected automatically.
        3. The solution is checked: it must be monotonic over the whole detector, and a warning is given if it is extrapolated far beyond the
           range covered by the lamp lines.

        In case you need references for the lamp peaks, here are some databases you can use (as of 12 September, 2024):
         * TNG DOLORES spectrograph reference lamps can be found in table 1 here: https://www.tng.iac.es/instruments/lrs/
         * SOAR Goodman Spectrograph Reference Lamp Library: https://soardocs.readthedocs.io/projects/lamps/en/latest/
         * Observatorio do Pico dos Dias (OPD-LNA) spectrographs: https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd/instrumentacao/instrumentos-e-detectores

        Parameters
        ----------
        lamp_peak_positions_list: list
            This is one of the outputs from function extraction.extracting(). It contains the positions (in pixels) of the peaks in the lamp
            spectrum of each trace.
        corresponding_wavelengths: list
            List with the wavelengths (in *Angstroms*) of the lamp lines, in the same order as the peaks. Use np.nan for peaks that you could not
            identify (they are ignored). A single list is used for all spectra; alternatively, give one list per spectrum.
        data_type: string
            Options are "target" or "std_star".
        poly_order: integer
            The order of the polynomial used to find the wavelength solution.
        diagnostic_plots: boolean
            If True, easyspec will plot the wavelength solution and fit residuals.
        clip_sigma: float
            The line with the largest residual is rejected if its residual is larger than clip_sigma times the RMS of the fit to the other lines,
            and this is repeated until no line is rejected. Default is 4.
        max_iterations: integer
            Maximum number of iterations of the sigma clipping. Default is 10.
        match_tolerance: float
            Maximum distance (in pixels) between the predicted and the detected position of a line, used when a spectrum does not have exactly
            one detected peak per wavelength. Default is 3.

        Returns
        -------
        wavelengths_list: list
            List with the wavelength solutions (astropy Quantity arrays in Angstroms, one value per pixel) for each given spectrum.
        wavelengths_fit_rms_list: list
            The RMS of the residuals of the lines used in each fit (corrected for the number of free parameters), in Angstroms. It measures the
            accuracy of the wavelength scale, and is saved in the output files as the systematic wavelength error.

        Notes
        -----
        The details of each solution (polynomial, lines used and rejected, residuals, dispersion in Angstrom/pixel, and the 1-sigma
        uncertainty of the solution at each pixel, from the covariance of the fit) are stored in the attribute self.wavelength_info (or
        self.std_wavelength_info for the standard star).
        """

        if self.image_shape is None:
            raise NameError("The image dimensions are unknown. Please load your data with the function extraction.import_data() first.")
        if data_type not in ("target", "std_star"):
            raise ValueError(f"Invalid data_type '{data_type}'. Options are 'target' or 'std_star'.")
        if not (isinstance(poly_order, (int, np.integer)) and poly_order >= 1):
            raise ValueError("'poly_order' must be an integer >= 1.")
        n_spectra = len(lamp_peak_positions_list)
        if n_spectra == 0:
            raise ValueError("'lamp_peak_positions_list' is empty.")

        # One list of wavelengths per spectrum:
        # (the elements are checked one by one: np.array() of a list with equal-length arrays would be 2D)
        if len(corresponding_wavelengths) > 0 and all(np.ndim(w) > 0 for w in corresponding_wavelengths):
            per_spectrum = [np.asarray(w, dtype=float) for w in corresponding_wavelengths]
            if len(per_spectrum) != n_spectra:
                raise ValueError(f"'corresponding_wavelengths' has {len(per_spectrum)} lists, but there are {n_spectra} lamp spectra.")
        else:
            per_spectrum = [np.asarray(corresponding_wavelengths, dtype=float)]*n_spectra
        positions_list = [np.asarray(p, dtype=float) for p in lamp_peak_positions_list]

        nx = self.image_shape[1]
        xrange = np.arange(nx)
        unit = u.AA

        def fit(x_lines, wavelengths_lines):
            """
            Polynomial fit with iterative rejection of the worst line (if its residual is larger than clip_sigma times the RMS of the other
            lines, which is robust even with few lines). Returns (polynomial, used, covariance of the coefficients, rms, residuals).
            """
            valid = np.isfinite(x_lines) & np.isfinite(wavelengths_lines)
            if np.count_nonzero(valid) < poly_order + 2:
                raise RuntimeError(f"Only {np.count_nonzero(valid)} identified lines, which is not enough for a polynomial of order {poly_order} "
                                   f"(at least {poly_order + 2} are needed). Identify more lines or use a lower 'poly_order'.")
            used = valid.copy()
            for _ in range(max_iterations):
                # A line is rejected only if at least 2 degrees of freedom remain afterwards (with fewer, the RMS of the other lines is
                # meaningless and a good line could be rejected):
                if np.count_nonzero(used) - 1 - (poly_order + 1) < 2:
                    break
                polynomial = np.polynomial.Polynomial.fit(x_lines[used], wavelengths_lines[used], poly_order, domain=[0, nx - 1])
                residuals = wavelengths_lines - polynomial(x_lines)
                candidates = np.nonzero(used)[0]
                worst = candidates[np.argmax(np.abs(residuals[candidates]))]
                # RMS of the fit without the worst line (leave-one-out), so that a large outlier does not inflate the threshold:
                others = used.copy()
                others[worst] = False
                polynomial_others = np.polynomial.Polynomial.fit(x_lines[others], wavelengths_lines[others], poly_order, domain=[0, nx - 1])
                residuals_others = wavelengths_lines - polynomial_others(x_lines)
                dof_others = np.count_nonzero(others) - (poly_order + 1)
                if dof_others < 1:
                    break
                scale = np.sqrt(np.sum(residuals_others[others]**2)/dof_others)
                if abs(residuals_others[worst]) > clip_sigma*max(scale, 1e-6):
                    used = others
                else:
                    break
            polynomial = np.polynomial.Polynomial.fit(x_lines[used], wavelengths_lines[used], poly_order, domain=[0, nx - 1])
            residuals = wavelengths_lines - polynomial(x_lines)
            dof = np.count_nonzero(used) - (poly_order + 1)
            rms = float(np.sqrt(np.sum(residuals[used]**2)/dof)) if dof > 0 else float(np.std(residuals[used]))
            # Covariance of the coefficients (in the normalized coordinate of the polynomial), scaled by the scatter of the residuals:
            t = polynomial.mapparms()[0] + polynomial.mapparms()[1]*x_lines[used]
            design = np.vander(t, poly_order + 1, increasing=True)
            covariance = np.linalg.pinv(design.T @ design)*rms**2
            return polynomial, used, covariance, rms, residuals

        def quality(result, x_lines, wavelengths_lines):
            """RMS in pixels and fraction of rejected lines (among the identified ones, i.e. ignoring np.nan) of a fit."""
            polynomial, used, _, rms, _ = result
            dispersion = np.median(np.abs(np.gradient(polynomial(xrange))))
            n_valid = np.count_nonzero(np.isfinite(x_lines) & np.isfinite(wavelengths_lines))
            return rms/dispersion, 1 - np.count_nonzero(used)/max(1, n_valid)

        # Pair the lines with the wavelengths. First, the spectra with exactly one peak per wavelength are paired in order; the pairing is
        # accepted only if the fit is good (RMS < 0.5 px, at most 20% of the lines rejected), since a missing line plus a spurious peak would
        # also give the right number of peaks. The best of these solutions is used as a reference to match the lines of the other spectra
        # by position (allowing for a small shift between the traces).
        paired, in_order_ok = [None]*n_spectra, [False]*n_spectra
        reference, reference_rms, fit_error = None, np.inf, None
        for number, (x_lines, wavelengths_lines) in enumerate(zip(positions_list, per_spectrum)):
            if len(x_lines) == len(wavelengths_lines):
                try:
                    result = fit(x_lines, wavelengths_lines)
                except RuntimeError as error:
                    fit_error = error
                    continue
                rms_pixels, rejected_fraction = quality(result, x_lines, wavelengths_lines)
                if rms_pixels < 0.5 and rejected_fraction <= 0.2:
                    paired[number], in_order_ok[number] = (x_lines, wavelengths_lines), True
                    if rms_pixels < reference_rms:
                        reference, reference_rms = result[0], rms_pixels

        for number, (x_lines, wavelengths_lines) in enumerate(zip(positions_list, per_spectrum)):
            if in_order_ok[number]:
                continue
            if reference is None and fit_error is not None:
                raise fit_error
            if reference is None:
                raise RuntimeError(f"The lamp lines of spectrum {number} could not be paired with the wavelengths: no spectrum has exactly one "
                                   "detected peak per wavelength with a good fit. Please check 'corresponding_wavelengths' (one value per peak, "
                                   "in the same order; use np.nan for peaks you cannot identify), or adjust 'lamp_peak_height'/'peak_distance' "
                                   "in extracting().")
            reference_wavelengths = reference(xrange)
            increasing = reference_wavelengths[-1] > reference_wavelengths[0]
            grid_w = reference_wavelengths if increasing else reference_wavelengths[::-1]
            grid_x = xrange if increasing else xrange[::-1]
            predicted = np.interp(wavelengths_lines, grid_w, grid_x, left=np.nan, right=np.nan)
            matched = np.full(len(wavelengths_lines), np.nan)
            offset = 0.0
            for _ in range(3):   # Iterate to measure the shift between this trace and the reference
                offsets = []
                for k, p in enumerate(predicted):
                    if np.isfinite(p) and len(x_lines) > 0:
                        nearest = np.argmin(np.abs(x_lines - (p + offset)))
                        if abs(x_lines[nearest] - (p + offset)) <= match_tolerance:
                            matched[k] = x_lines[nearest]
                            offsets.append(x_lines[nearest] - p)
                        else:
                            matched[k] = np.nan
                if len(offsets) == 0:
                    break
                offset = float(np.median(offsets))
            n_missing = np.count_nonzero(np.isfinite(wavelengths_lines) & ~np.isfinite(matched))
            print(f"Spectrum {number}: the lamp lines were matched by position to the solution of another spectrum (shift = {offset:+.2f} px; "
                  f"{n_missing} line(s) not found within {match_tolerance} px).")
            paired[number] = (matched, wavelengths_lines)

        wavelengths_list, wavelengths_fit_rms_list, info_list = [], [], []
        for number, (x_lines, wavelengths_lines) in enumerate(paired):
            polynomial, used, covariance, rms, residuals = fit(x_lines, wavelengths_lines)
            wavelengths = polynomial(xrange)
            # Uncertainty of the solution at each pixel (from the covariance of the coefficients):
            t = polynomial.mapparms()[0] + polynomial.mapparms()[1]*xrange
            design = np.vander(t, poly_order + 1, increasing=True)
            solution_error = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))
            dispersion = np.gradient(wavelengths)

            # Checks:
            if not (np.all(dispersion > 0) or np.all(dispersion < 0)):
                raise RuntimeError(f"The wavelength solution of spectrum {number} is not monotonic over the detector (it turns back near x = "
                                   f"{xrange[np.argmin(np.abs(dispersion))]}). Use a lower 'poly_order', or add lamp lines near the ends of the spectrum.")
            x_used = x_lines[used]
            if np.count_nonzero(used) - (poly_order + 1) < 3:
                warnings.warn(f"Spectrum {number}: only {np.count_nonzero(used)} lines for a polynomial of order {poly_order} ({np.count_nonzero(used) - poly_order - 1} "
                              "degree(s) of freedom), so the RMS of the residuals is a poor estimate of the accuracy of the solution, and outliers "
                              "cannot be identified. Identify more lamp lines, or use a lower 'poly_order'.")
            if x_used.min() > 0.15*nx or x_used.max() < 0.85*nx:
                warnings.warn(f"Spectrum {number}: the lamp lines used cover only x = {x_used.min():.0f}-{x_used.max():.0f} of the {nx} pixels of "
                              "the detector, so the wavelength solution is extrapolated beyond this range (and less reliable there).")
            rejected = np.isfinite(x_lines) & np.isfinite(wavelengths_lines) & ~used

            wavelengths_list.append(wavelengths*unit)
            wavelengths_fit_rms_list.append(rms)
            info_list.append({"polynomial": polynomial, "lines_x": x_lines, "lines_wavelength": wavelengths_lines, "used": used,
                              "residuals": residuals, "rms": rms, "rejected_wavelengths": wavelengths_lines[rejected],
                              "dispersion": dispersion*unit, "solution_error": solution_error*unit, "x_range_lines": (x_used.min(), x_used.max())})

            print(f"Spectrum {number}: {np.count_nonzero(used)} lines used, {np.count_nonzero(rejected)} rejected"
                  + (f" ({', '.join(f'{w:.2f}' for w in wavelengths_lines[rejected])} A)" if np.any(rejected) else "")
                  + f"; RMS = {rms:.4f} A ({rms/np.median(np.abs(dispersion)):.3f} px); dispersion = {np.median(np.abs(dispersion)):.4f} A/px; "
                  f"lambda = {wavelengths.min():.1f}-{wavelengths.max():.1f} A.")

            if diagnostic_plots:
                plt.figure(figsize=(12,5))
                plt.plot(x_lines[used], wavelengths_lines[used], 'o', label="Lamp lines used")
                if np.any(rejected):
                    plt.plot(x_lines[rejected], wavelengths_lines[rejected], 'x', color="red", ms=9, label="Rejected lines")
                plt.plot(xrange, wavelengths, '-', label="Wavelength solution")
                plt.ylabel(r"$\lambda(x)$" + f" [{str(unit)}]")
                plt.xlabel("x (pixels)")
                plt.title(f"Wavelength solution - spectrum {number}")
                plt.grid(linestyle=":",which="both")
                plt.minorticks_on()
                plt.legend()

                plt.figure(figsize=(12,5))
                plt.errorbar(x_lines[used], residuals[used], fmt='o', label=f"Lines used (RMS = {rms:.4f} {unit})")
                if np.any(rejected):
                    plt.plot(x_lines[rejected], residuals[rejected], 'x', color="red", ms=9, label="Rejected lines")
                plt.fill_between(xrange, -solution_error, solution_error, color="C1", alpha=0.3, label=r"$\pm1\sigma$ uncertainty of the solution")
                plt.axhline(0, color="k", linestyle="--")
                plt.ylabel(f"Wavelength residuals [{str(unit)}]")
                plt.xlabel("x (pixels)")
                plt.title(f"Fit residuals - spectrum {number}")
                plt.grid(linestyle=":",which="both")
                plt.minorticks_on()
                plt.legend()

        if diagnostic_plots:
            plt.show()

        if data_type == "target":
            self.wavelengths_fit_rms_list = copy.deepcopy(wavelengths_fit_rms_list)
            self.wavelength_info = info_list
        else:
            self.std_wavelengths_fit_rms_list = copy.deepcopy(wavelengths_fit_rms_list)
            self.std_wavelength_info = info_list

        return wavelengths_list, wavelengths_fit_rms_list

    def _load_extinction_curve(self, observatory, custom_observatory):

        """Return (wavelength [Angstrom], extinction [mag/airmass], label) of the extinction curve, sorted by wavelength."""

        if custom_observatory is not None:
            try:
                table = np.loadtxt(custom_observatory, ndmin=2)
            except Exception as error:
                raise RuntimeError(f"The custom_observatory file {custom_observatory} could not be read: {error}")
            if table.shape[1] != 2:
                raise RuntimeError("The custom_observatory file does not have two columns. Please be sure to use a txt file with two columns, the "
                                   "first with the wavelengths (in Angstroms) and the second with the extinction in mag/airmass.")
            label = f"custom ({Path(custom_observatory).name})"
        else:
            available = sorted(p.name.replace("_airmass_extinction.txt", "") for p in libpath.glob("*_airmass_extinction.txt"))
            if observatory not in available:
                raise ValueError(f"Unknown observatory '{observatory}'. Options are: " + ", ".join(f"'{a}'" for a in available)
                                 + ". You can also give your own extinction curve with 'custom_observatory'.")
            table = np.loadtxt(libpath / f"{observatory}_airmass_extinction.txt", ndmin=2)
            label = observatory
        table = table[np.all(np.isfinite(table), axis=1)]
        order = np.argsort(table[:, 0])
        wavelength, extinction = table[order, 0], table[order, 1]
        if len(wavelength) < 2 or np.any(np.diff(wavelength) <= 0):
            raise RuntimeError(f"The extinction curve ({label}) must have at least two points with distinct wavelengths.")
        return wavelength, extinction, label

    @_plot_style
    def extinction_correction(self, spec_list, wavelengths_list, observatory, data_type, custom_observatory = None, spline_order = 1, plots = True):

        """
        This function removes the atmospheric extinction from the observed spectra:

            F_corrected(lambda) = F_observed(lambda) * 10^(0.4 * k(lambda) * X),

        where k(lambda) is the extinction curve of the observatory (in mag/airmass) and X is the airmass of the observation (read by
        import_data()). The variance of the spectra (computed by extracting()) is corrected by the square of the same factor.

        Parameters
        ----------
        spec_list: list
            List with the extracted spectra of all the targets in the image or only the brightest spectrum. This is one of the outputs from function extraction.extracting().
        wavelengths_list: list
            List with the wavelength solutions for each given spectrum. This is the main result from function extraction.wavelength_calibration().
        observatory: string
            This is the observatory where you collected your data. Options are "ctio", "kpno", "lapalma", "mko", "mtham", "paranal", "apo".
        data_type: string
            Options are "target" or "std_star". We use this information to correctly select the airmass for the extinction correction.
        custom_observatory: string
            Optional. Path to a txt file containing the airmass extinction curve. This file must have two columns, the first with the wavelengths
            *in Angstroms* and the second with the extinction in mag/airmass.
        spline_order: integer
            Order of the interpolation of the extinction curve: 1 (default) for linear interpolation, or 3 for a cubic spline (only for well
            sampled curves). The curve is never extrapolated: beyond its range, its first/last value is used, with a warning.
        plots: boolean
            If True, easyspec will plot a comparison between the corrected and original spectrum, as well as the extinction interpolation curve.

        Returns
        -------
        spec_atm_corrected_list: list
            List with all the spectra corrected for atmospheric extinction.

        Notes
        -----
        The corrected variances are stored in self.spec_atm_variance_list (target) or self.std_spec_atm_variance_list (standard star). The
        correction factors are stored in self.extinction_factor_list (or self.std_extinction_factor_list). The uncertainty of the extinction
        curve itself is not included (it is a systematic error, common to the target and the standard star, which largely cancels out in the
        flux calibration when both are observed at similar airmasses).
        """

        reference_dictionary = {"ctio" : "The CTIO extinction curve was originally distributed with IRAF and comes from the work of Stone & Baldwin (1983 MN 204, 347) \
plus Baldwin & Stone (1984 MN 206, 241). The first of these papers lists the points from 3200-8370A while the second extended the \
flux calibration from 6056 to 10870A but the derived extinction curve was not given in the paper. The IRAF table follows SB83 out \
to 6436, the redder points presumably come from BS84 with averages used in the overlap region. More recent CTIO extinction curves \
are shown as Figures in Hamuy et al (92, PASP 104, 533 ; 94 PASP 106, 566).",
"kpno" : "The KPNO extinction table was originally distributed with IRAF. The usage of this table is discouraged since the data provenance of this data is unclear.",
"lapalma" : "Extinction table for Roque de Los Muchachos Observatory, La Palma. Described in https://www.ing.iac.es/Astronomy/observing/manuals/ps/tech_notes/tn031.pdf",
"mko" : "Median atmospheric extinction data for Mauna Kea Observatory measured by the Nearby SuperNova Factory: https://www.aanda.org/articles/aa/pdf/2013/01/aa19834-12.pdf",
"mtham" : "Extinction table for Lick Observatory on Mt. Hamilton constructed from https://mthamilton.ucolick.org/techdocs/standards/lick_mean_extinct.html",
"paranal": "Updated extinction table for ESO-Paranal taken from https://www.aanda.org/articles/aa/pdf/2011/03/aa15537-10.pdf",
"apo" : "Extinction table for Apache Point Observatory. Based on the extinction table used for SDSS and available \
at https://www.apo.nmsu.edu/arc35m/Instruments/DIS/ (https://www.apo.nmsu.edu/arc35m/Instruments/DIS/images/apoextinct.dat)."}

        # ---------------------------------------------------------------- Input checks
        if data_type == "target":
            airmass = self.airmass_target
            variance_list = getattr(self, "spec_variance_list", None)
        elif data_type == "std_star":
            airmass = self.airmass_std_star
            variance_list = getattr(self, "std_spec_variance_list", None)
            if len(spec_list) > 1:
                raise RuntimeError(f"We have {len(spec_list)} spectra given as input. Since the variable data_type was set as 'std_star', the variable "
                                   "'spec_list' must contain only one spectrum.")
        else:
            raise ValueError("Invalid entry for data_type. The options are 'target' or 'std_star'.")
        if airmass is None:
            raise RuntimeError(f"The airmass of the {'target' if data_type == 'target' else 'standard star'} is unknown. Please give it in "
                               "extraction.import_data() (airmass_target / airmass_std_star, or the header entry airmass_header_entry).")
        airmass = float(airmass)
        if not 1.0 <= airmass <= 4.0:
            warnings.warn(f"Unusual airmass = {airmass} (the airmass is >= 1, and rarely larger than ~3). Please check its value.")
        if len(spec_list) != len(wavelengths_list):
            raise ValueError(f"'spec_list' has {len(spec_list)} spectra, but 'wavelengths_list' has {len(wavelengths_list)} wavelength solutions.")
        if spline_order not in (1, 3):
            raise ValueError("'spline_order' must be 1 (linear interpolation) or 3 (cubic spline).")
        if variance_list is not None and len(variance_list) != len(spec_list):
            variance_list = None   # The stored variances do not correspond to these spectra

        extinction_wavelength, extinction, label = self._load_extinction_curve(observatory, custom_observatory)
        if spline_order == 3 and len(extinction_wavelength) >= 4:
            curve = interpolate.CubicSpline(extinction_wavelength, extinction, extrapolate=False)
        else:
            curve = lambda w: np.interp(w, extinction_wavelength, extinction, left=np.nan, right=np.nan)

        spec_atm_corrected_list, corrected_variance_list, factor_list = [], [], []
        for number, (spec, wavelength) in enumerate(zip(spec_list, wavelengths_list)):
            wavelength_values = wavelength.to(u.AA).value if isinstance(wavelength, u.Quantity) else np.asarray(wavelength, dtype=float)
            spec = np.asarray(spec, dtype=float)
            if spec.shape != wavelength_values.shape:
                raise ValueError(f"Spectrum {number} has {spec.size} pixels, but its wavelength solution has {wavelength_values.size}.")
            k = curve(wavelength_values)
            outside = ~np.isfinite(k)
            if np.any(outside):
                k = np.where(wavelength_values < extinction_wavelength[0], extinction[0], k)
                k = np.where(wavelength_values > extinction_wavelength[-1], extinction[-1], k)
                warnings.warn(f"Spectrum {number}: the extinction curve ({label}) covers {extinction_wavelength[0]:.0f}-{extinction_wavelength[-1]:.0f} A, "
                              f"but the spectrum goes from {np.nanmin(wavelength_values):.0f} to {np.nanmax(wavelength_values):.0f} A. Outside the curve, "
                              "its first/last value was used.")
            factor = 10**(0.4*k*airmass)
            spec_atm_corrected = spec*factor
            spec_atm_corrected_list.append(spec_atm_corrected)
            factor_list.append(factor)
            if variance_list is not None:
                corrected_variance_list.append(np.asarray(variance_list[number], dtype=float)*factor**2)

            if plots:
                plt.figure(figsize=(12,5))
                plt.plot(wavelength_values, spec, label="No atm correction", linewidth=0.8)
                plt.plot(wavelength_values, spec_atm_corrected, color="orange", label="Atm corrected", linewidth=0.8)
                plt.xlabel(f"Wavelength [{u.AA}]")
                plt.ylabel("Counts")
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                if data_type == "target":
                    plt.title(f"Atmospheric-corrected spectrum {number} - "+str(self.target_name)+f" field - Airmass = {airmass:.3f}")
                else:
                    plt.title(f"Atmospheric-corrected spectrum for the standard star - Airmass = {airmass:.3f}")
                plt.legend()

                plt.figure(figsize=(12,5))
                in_range = (extinction_wavelength >= np.nanmin(wavelength_values) - 500) & (extinction_wavelength <= np.nanmax(wavelength_values) + 500)
                plt.plot(extinction_wavelength[in_range], extinction[in_range], 'o', ms=4, label=f"Extinction data ({label})")
                plt.plot(wavelength_values, k, label="Interpolation used")
                plt.axhline(y=0.0, color='r', linestyle=':')
                plt.xlabel(f"Wavelength [{u.AA}]")
                plt.ylabel("Extinction (mag/airmass)")
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                if data_type == "target":
                    plt.title(f"Extinction curve for spectrum {number} - "+str(self.target_name)+" field")
                else:
                    plt.title("Extinction curve for the standard star")
                plt.legend()

        if plots:
            plt.show()

        # ---------------------------------------------------------------- Store the results
        # The corrected uncertainties are always recomputed from the variances of extracting(), so calling this function more than once does
        # not apply the correction twice.
        if data_type == "target":
            self.extinction_factor_list = factor_list
            if variance_list is not None:
                self.spec_atm_variance_list = corrected_variance_list
                self.spec_systematic_error_list = [np.sqrt(v) for v in corrected_variance_list]   # Used by target_flux_calibration()
            elif self.spec_systematic_error_list is not None and len(self.spec_systematic_error_list) == len(factor_list):
                # Uncertainties from a previous easyspec version (no variance available): scaled once
                if not getattr(self, "_legacy_errors_extinction_corrected", False):
                    self.spec_systematic_error_list = [e*f for e, f in zip(self.spec_systematic_error_list, factor_list)]
                    self._legacy_errors_extinction_corrected = True
        else:
            self.std_extinction_factor_list = factor_list
            if variance_list is not None:
                self.std_spec_atm_variance_list = corrected_variance_list

        if custom_observatory is None:
            print("Please cite the work where this extinction curve was measured:\n"+reference_dictionary.get(observatory, observatory))

        return spec_atm_corrected_list


    def list_available_standards(self, std_star_dataset = None, star_name = None):

        """
        This function lists the available datasets of standard stars, the stars of a dataset, or searches all datasets for a star.

        Parameters
        ----------
        std_star_dataset: string
            If None (and star_name is None), the function prints all available datasets. Options are "ctiocal", "ctionewcal", "irscal",
            "bstdscal", "spec16cal", "spechayescal", "iidscal", "spec50cal", "redcal", "oke1990".
        star_name: string
            Optional. Name of your standard star, e.g. "LTT 4364", "Feige 110", "BD+28 4211", "EG 21" (spaces, signs and case do not
            matter; usually it is the OBJECT entry of the header of the standard star file). The function then lists all archival files of
            this star, in all datasets, with their wavelength ranges.

        Returns
        -------
        If std_star_dataset is given: (list of the files of the dataset, reference of the dataset).
        If star_name is given: list of (dataset, file name, star name, first wavelength, last wavelength).
        Otherwise: list of the available datasets.
        """

        reference_dictionary = {"blackbody" :  "Blackbody flux distributions in various magnitude bands (templates; they cannot be used for the absolute flux calibration).",
"bstdscal" :  "The brighter KPNO IRS standards (i.e. those with HR numbers) at 29 bandpasses, data from various\
 sources transformed to the Hayes and Latham system, unpublished.",
"ctiocal" :  "Fluxes for the southern tertiary standards as published by Baldwin & Stone, 1984, MNRAS,\
 206, 241 and Stone and Baldwin, 1983, MNRAS, 204, 347.",
"ctionewcal" : "Fluxes at 50A steps in the blue range 3300-7550A for the tertiary standards of Baldwin and\
 Stone derived from the revised calibration of Hamuy et al., 1992, PASP, 104, 533. This dataset also contains the fluxes of\
 the tertiaries in the red (6050-10000A) at 50A steps as published in Hamuy et al. 1994, PASP, 106, 566. The combined fluxes are\
 obtained by gray shifting the blue fluxes to match the red fluxes in the overlap region of 6500A-7500A and averaging the red and\
 blue fluxes in the overlap. The separate red and blue fluxes may be selected by following the star name with 'red' or 'blue'; i.e. cd32blue.dat.",
"iidscal" : "Dataset of the KPNO IIDS standards at 29 bandpasses, data from various sources transformed to the Hayes and Latham system, unpublished.",
"irscal" :  "Dataset of the KPNO IRS standards at 78 bandpasses, data from various sources transformed to the Hayes and Latham\
 system, unpublished (note that in this dataset the brighter standards have no values - the 'bstdscal' dataset must be used for these standards).",
"oke1990" : "Dataset of spectrophotometric standards observed for use with the HST, Table VII, Oke 1990, AJ, 99, 1621 (no correction\
 was applied). An arbitrary 1A bandpass is specified for these smoothed and interpolated flux 'points'.",
"redcal" : "Dataset of standard stars with flux data beyond 8370A. These stars are from the IRS or the IIDS dataset but have data\
 extending as far out into the red as the literature permits. Data from various sources.",
"spechayescal" : "The KPNO spectrophotometric standards at the Hayes flux points, Table IV, Spectrophotometric Standards, Massey et al., 1988, ApJ 328, p. 315.",
"spec16cal" : "Dataset containing fluxes at 16A steps in the blue range 3300-7550A for the secondary standards, published in Hamuy et al., 1992, PASP, 104, 533.\
 This dataset also contains the fluxes of the secondaries in the red (6020-10300A) at 16A steps as published in Hamuy et al. 1994, PASP, 106, 566.\
 The combined fluxes are obtained by gray shifting the blue fluxes to match the red fluxes in the overlap region of 6500A-7500A and averaging the blue\
 and red fluxes in the overlap. The separate red and blue fluxes may be selected by following the star name with 'red' or 'blue'; i.e. hr1544blue.dat.",
"spec50cal" : "The KPNO spectrophotometric standards at 50 A intervals. The data are from (1) Table V, Spectrophotometric Standards, Massey et al., 1988,\
 ApJ 328, p. 315 and (2) Table 3, The Kitt Peak Spectrophotometric Standards: Extension to 1 micron, Massey and Gronwall, 1990, ApJ 358, p. 344."}

        available_dataset_names = sorted(p.name for p in libpath_std.iterdir() if p.is_dir())

        if star_name is not None:
            matches = flux_calibration.find_standard_files(star_name, libpath_std)
            if not matches:
                print(f"No archival file found for '{star_name}'. Try another spelling (e.g. 'LTT 4364' or 'L4364'), or list the stars of a "
                      "dataset with extraction.list_available_standards(std_star_dataset='ctionewcal').")
                return []
            print(f"Archival files for '{star_name}':")
            for dataset, file_name, name, first, last in matches:
                print(f"    std_star_dataset = '{dataset}', std_star_archive_file = '{file_name}'  ({name}; {first:.0f}-{last:.0f} A)")
            print("Prefer a file that covers the whole wavelength range of your spectra (e.g. the combined blue+red files of ctionewcal/spec16cal).")
            return matches

        if std_star_dataset is None:
            print("Available datasets:")
            for name in available_dataset_names:
                print(f"    {name}: {reference_dictionary.get(name, '')[:110]}...")
            return available_dataset_names

        if std_star_dataset in available_dataset_names:
            std_stars = sorted(p.name for p in (libpath_std / std_star_dataset).glob("*.dat"))
            return std_stars, reference_dictionary.get(std_star_dataset, "")

        raise ValueError(f"Dataset '{std_star_dataset}' not found in our library. Available datasets: {available_dataset_names}.")

    @_plot_style
    def std_star_normalization(self, spec_atm_corrected_std, wavelengths_std, std_star_dataset, std_star_archive_file, smooth_window = None,
                               exclude_regions = None, smooth_window_archive = None, interpolation_order = None, plots = True,
                               mask_telluric = True, knot_spacing = None, clip_sigma = 3.5, archive_uncertainty = 0.01, archive_unit = None):

        """
        This function computes the sensitivity function of the instrument, i.e. the conversion from counts/s/Angstrom to flux density
        (erg/cm2/s/Angstrom), by comparing the observed standard star spectrum with its archival spectrum.

        The archival spectra give the flux of the star averaged over bandpasses (typically 16-50 Angstroms wide). The observed spectrum
        (normalized by the exposure time and by the dispersion, i.e. in counts/s/Angstrom) is averaged over exactly the same bandpasses, so
        the comparison does not depend on the spectral resolution, and the absorption lines of the star do not need to be masked. The
        sensitivity points (one per bandpass) are fitted with a smooth curve (cubic spline, in magnitudes), weighted by their uncertainties
        and with robust sigma clipping; the uncertainty of the curve is obtained from the covariance of the fit.

        Parameters
        ----------
        spec_atm_corrected_std: list or array
            The standard star spectrum corrected for atmospheric extinction (output of extraction.extinction_correction()).
        wavelengths_std: list or array
            The wavelength solution of the standard star (output of extraction.wavelength_calibration()).
        std_star_dataset: string
            The dataset of your standard star: "ctiocal", "ctionewcal", "irscal", "bstdscal", "spec16cal", "spechayescal", "iidscal",
            "spec50cal", "redcal", "oke1990". Use extraction.list_available_standards(star_name="your star") to find it. If None,
            'std_star_archive_file' must be the path to your own archival file.
        std_star_archive_file: string
            The name of the archival file of the standard star, e.g. "l4364.dat". Alternatively (with std_star_dataset = None), the path to
            your own file, with columns: wavelength (Angstrom), AB magnitude (or flux density, see 'archive_unit') and, optionally, the width
            of the bandpasses (Angstrom).
        smooth_window, smooth_window_archive, interpolation_order:
            Deprecated and ignored (they were used by the median-filter method of previous easyspec versions). Use 'knot_spacing' to control
            the smoothness of the sensitivity curve.
        exclude_regions: list
            Optional. Regions (in Angstroms) to be excluded from the fit, e.g.: exclude_regions = [[4000,4100],[8000,8500]]. The bandpasses
            that overlap these regions are not used.
        plots: boolean
            If True, easyspec plots the sensitivity points and curve, its residuals, and the calibrated standard star compared to its
            archival spectrum.
        mask_telluric: boolean
            If True (default), the bandpasses overlapping the telluric absorption bands (O2 B and A bands at ~6870 and ~7600 A, and the
            strongest water bands) are excluded, since their depth changes with time and airmass. The sensitivity curve is interpolated
            across them. Note that the telluric absorption is not removed from the target spectra.
        knot_spacing: float
            Distance between the knots of the spline, in Angstroms. Smaller values follow sharper features of the sensitivity (e.g. near a
            dichroic) but are noisier. Default: the largest of 4x the spacing of the archival points, 1/20 of the wavelength range, and 150 A.
        clip_sigma: float
            Points that deviate more than clip_sigma times their uncertainty (scaled by the scatter of the residuals) are rejected. Default 3.5.
        archive_uncertainty: float
            Relative uncertainty of the archival fluxes, added in quadrature to the statistical uncertainty of each point. Default: 0.01 (1%).
        archive_unit: astropy unit
            Only for your own files: the unit of the second column (default: u.ABmag; e.g. u.erg/u.cm**2/u.s/u.AA or u.mJy).

        Returns
        -------
        sensitivity_function: easyspec.extraction.flux_calibration.SensitivityFunction
            The flux calibration solution. Calling it with wavelengths in Angstroms, sensitivity_function(wavelengths), returns the
            sensitivity in (erg/cm2/s/Angstrom)/(counts/s/Angstrom); sensitivity_function.relative_error(wavelengths) gives its 1-sigma
            relative uncertainty. Pass it to extraction.target_flux_calibration(). It is also stored in self.sensitivity_function.

        Notes
        -----
        The absolute flux scale assumes that the slit losses of the target and of the standard star are equal (e.g. the standard star was
        observed with a wide slit and the target with the same slit, under similar seeing). Otherwise, the shape of the calibrated spectrum
        is still correct, but its normalization may be off by a (nearly) wavelength-independent factor.
        """

        for name, value in (("smooth_window", smooth_window), ("smooth_window_archive", smooth_window_archive),
                            ("interpolation_order", interpolation_order)):
            if value is not None:
                warnings.warn(f"'{name}' is deprecated and ignored: the sensitivity is now fitted with a spline over the bandpasses of the "
                              "archival spectrum. Use 'knot_spacing' to control its smoothness.", FutureWarning)

        # ---------------------------------------------------------------- Inputs
        spectrum = spec_atm_corrected_std
        if isinstance(spectrum, (list, tuple)):
            if len(spectrum) != 1:
                raise ValueError(f"'spec_atm_corrected_std' has {len(spectrum)} spectra; only one standard star spectrum is expected.")
            spectrum = spectrum[0]
        wavelength = wavelengths_std
        if isinstance(wavelength, (list, tuple)):
            wavelength = wavelength[0]
        wavelength = wavelength.to(u.AA).value if isinstance(wavelength, u.Quantity) else np.asarray(wavelength, dtype=float)
        spectrum = np.asarray(getattr(spectrum, "value", spectrum), dtype=float)
        if spectrum.shape != wavelength.shape:
            raise ValueError(f"The standard star spectrum has {spectrum.size} pixels, but its wavelength solution has {wavelength.size}.")
        if self.exposure_std_star is None or not float(self.exposure_std_star) > 0:
            raise RuntimeError("The exposure time of the standard star is unknown. Please give it in extraction.import_data() (exposure_std_star, "
                               "or the header entry exposure_header_entry).")
        exposure = float(self.exposure_std_star)

        def stored(name):
            values = getattr(self, name, None)
            if values is not None and len(values) == 1 and np.shape(values[0]) == spectrum.shape:
                return np.asarray(values[0], dtype=float)
            return None
        variance = stored("std_spec_atm_variance_list")
        if variance is None:
            variance = stored("std_spec_variance_list")
            if variance is not None and getattr(self, "std_extinction_factor_list", None) is not None:
                variance = variance*np.asarray(self.std_extinction_factor_list[0])**2
        mask = stored("std_spec_mask_list")
        bad = (mask.astype(int) & (1 | 4)) > 0 if mask is not None else np.zeros(spectrum.shape, dtype=bool)
        bad |= ~np.isfinite(spectrum)

        # ---------------------------------------------------------------- Archival spectrum
        if std_star_dataset is None:
            archive_path = Path(std_star_archive_file)
        else:
            archive_path = libpath_std / std_star_dataset / std_star_archive_file
            if not archive_path.suffix:
                archive_path = archive_path.with_suffix(".dat")
        if not archive_path.exists():
            suggestions = flux_calibration.find_standard_files(Path(std_star_archive_file).stem, libpath_std)
            hint = ("Files found for this star: " + ", ".join(f"{d}/{f}" for d, f, *_ in suggestions[:8]) + ".") if suggestions else \
                   "Use extraction.list_available_standards(star_name='your star') to find the archival file of your standard star."
            raise FileNotFoundError(f"Archival file {archive_path} not found. {hint}")
        # Is it the right star? The name in the header of the standard star image (OBJECT) is compared with the archival file.
        header = getattr(self, "std_star_header", None)
        object_name = str(header.get("OBJECT", "")).strip() if header is not None else ""
        if object_name and std_star_dataset is not None:
            matches = flux_calibration.find_standard_files(object_name, libpath_std)
            if matches and not any(d == std_star_dataset and f == archive_path.name for d, f, *_ in matches):
                warnings.warn(f"The header of the standard star image says OBJECT = '{object_name}', but the archival file "
                              f"{std_star_dataset}/{archive_path.name} is not one of the files of this star ("
                              + ", ".join(f"{d}/{f}" for d, f, *_ in matches[:6]) + "). A wrong archival spectrum still gives a smooth "
                              "sensitivity curve, but a wrong flux calibration. Please check.")
        archive_wavelength, archive_flux, archive_width = flux_calibration.read_standard_file(
            archive_path, column_unit=u.ABmag if archive_unit is None else archive_unit)

        # ---------------------------------------------------------------- Sensitivity function
        sensitivity = flux_calibration.compute_sensitivity(wavelength, spectrum/exposure, None if variance is None else variance/exposure**2, bad,
                                                           archive_wavelength, archive_flux, archive_width, exclude_regions=exclude_regions,
                                                           mask_telluric=mask_telluric, knot_spacing=knot_spacing, clip_sigma=clip_sigma,
                                                           archive_uncertainty=archive_uncertainty)
        info = sensitivity.info
        info["archive_file"] = str(archive_path)
        used, candidates = info["used"], info["candidates"]
        n_rejected = np.count_nonzero(candidates & ~used)
        relative_error = sensitivity.relative_error(wavelength[sensitivity.in_range(wavelength)])
        print(f"Sensitivity function: {np.count_nonzero(used)} bandpasses used, {n_rejected} rejected, "
              f"{np.count_nonzero(info['observed'] & info['excluded'])} excluded (telluric/excluded regions); spline with {info['n_knots']} interior "
              f"knots (spacing {info['knot_spacing']:.0f} A). Residual RMS = {info['rms_percent']:.1f}%; calibration uncertainty "
              f"(median) = {100*np.median(relative_error):.1f}%; valid range = {sensitivity.wavelength_range[0]:.0f}-{sensitivity.wavelength_range[1]:.0f} A.")
        if info["extrapolated_range"]:
            ranges = ", ".join(f"{a:.0f}-{b:.0f} A" for a, b in info["extrapolated_range"])
            warnings.warn(f"The bandpasses at the edges of the spectrum ({ranges}) are excluded from the fit (telluric bands or "
                          "exclude_regions), so the sensitivity function is extrapolated linearly there (with a larger uncertainty). Check the plots; use mask_telluric=False or exclude_regions "
                          "to change this.")
        if n_rejected > 0.2*np.count_nonzero(candidates):
            warnings.warn(f"{n_rejected} of {np.count_nonzero(candidates)} bandpasses were rejected. Check the plots: the wavelength solution of the "
                          "standard star, the archival file (is it the right star?), or regions to exclude (e.g. a bad column).")
        if info["rms_percent"] > 5:
            warnings.warn(f"The residuals of the sensitivity fit are large (RMS = {info['rms_percent']:.1f}%). Check the plots; you may need a "
                          "smaller 'knot_spacing' or to exclude some regions.")
        band_width = np.max(info["band_width"][info["observed"]])
        lost = [(a, b) for a, b in ((wavelength.min(), sensitivity.wavelength_range[0]), (sensitivity.wavelength_range[1], wavelength.max()))
                if b - a > band_width]
        if lost:
            warnings.warn(f"The sensitivity function covers {sensitivity.wavelength_range[0]:.0f}-{sensitivity.wavelength_range[1]:.0f} A. The parts of "
                          "the spectra in " + ", ".join(f"{a:.0f}-{b:.0f} A" for a, b in lost) + " cannot be calibrated with this archival file (it "
                          "has no bandpass fully inside them) and will be removed from the target spectra. Use an archival file with a wider "
                          "coverage if possible (see extraction.list_available_standards(star_name=...)).")

        # ---------------------------------------------------------------- Plots
        if plots:
            band = info["band_wavelength"]
            observed = info["observed"]
            rejected = candidates & ~used
            excluded = observed & info["excluded"]
            dispersion = np.abs(np.gradient(wavelength))
            grid = np.linspace(*sensitivity.wavelength_range, 1000)

            plt.figure(figsize=(12,5))
            plt.plot(wavelength, spectrum/exposure/dispersion, color="k", linewidth=0.6, label="Observed standard star")
            plt.plot(band[used], info["observed_density"][used], "o", color="C0", ms=4, label="Mean over the archival bandpasses (used)")
            if np.any(rejected):
                plt.plot(band[rejected], info["observed_density"][rejected], "x", color="red", ms=8, label="Rejected")
            if np.any(excluded):
                plt.plot(band[excluded], info["observed_density"][excluded], "s", mfc="none", color="gray", ms=6, label="Excluded (telluric/regions)")
            for low, high in (flux_calibration.TELLURIC_BANDS if mask_telluric else []):
                plt.axvspan(low, high, color="gray", alpha=0.12)
            plt.xlabel(r"Wavelength [$\AA$]")
            plt.ylabel(r"Counts s$^{-1}$ $\AA^{-1}$")
            plt.xlim(wavelength.min(), wavelength.max())
            plt.title("Standard star: observed spectrum (atmospheric-corrected)")
            plt.minorticks_on()
            plt.grid(which="both", linestyle=":")
            plt.legend()

            figure, (top, bottom) = plt.subplots(2, 1, figsize=(12,7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            magnitude, magnitude_error = info["magnitude"], info["magnitude_error"]
            curve, curve_error = sensitivity.magnitude(grid), sensitivity.magnitude_error(grid)
            top.errorbar(band[used], magnitude[used], magnitude_error[used], fmt="o", color="C0", ms=4, label="Sensitivity points (used)")
            if np.any(rejected):
                top.plot(band[rejected], magnitude[rejected], "x", color="red", ms=8, label="Rejected")
            if np.any(excluded):
                top.plot(band[excluded], magnitude[excluded], "s", mfc="none", color="gray", ms=6, label="Excluded")
            top.plot(grid, curve, color="C1", label="Spline fit")
            top.fill_between(grid, curve - curve_error, curve + curve_error, color="C1", alpha=0.3, label=r"$\pm1\sigma$")
            top.set_ylabel(r"2.5 log$_{10}$ S")
            top.set_title(f"Sensitivity function ({archive_path.name})")
            top.grid(which="both", linestyle=":")
            top.minorticks_on()
            top.legend()
            percent = 100*0.4*np.log(10)
            shown = observed & np.isfinite(info["residuals"])
            bottom.errorbar(band[used], percent*info["residuals"][used], percent*magnitude_error[used], fmt="o", color="C0", ms=4)
            bottom.plot(band[shown & ~used], percent*info["residuals"][shown & ~used], "x", color="gray", ms=6)
            bottom.fill_between(grid, -percent*curve_error, percent*curve_error, color="C1", alpha=0.3)
            bottom.axhline(0, color="k", linestyle="--")
            residual_limit = 4*max(1.0, info["rms_percent"])
            bottom.set_ylim(-residual_limit, residual_limit)
            bottom.set_ylabel("Residuals [%]")
            bottom.set_xlabel(r"Wavelength [$\AA$]")
            bottom.grid(which="both", linestyle=":")
            bottom.minorticks_on()

            figure, (top, bottom) = plt.subplots(2, 1, figsize=(12,7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            in_range = sensitivity.in_range(wavelength)
            calibrated = np.where(in_range, spectrum/exposure/dispersion*sensitivity(wavelength).value, np.nan)
            top.plot(wavelength, calibrated, color="orange", linewidth=0.7, label="Calibrated standard star")
            top.plot(band[observed], info["archive_flux"][observed], "o", color="C0", ms=4, label="Archival spectrum")
            top.set_ylabel(r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]")
            top.set_ylim(0, 1.2*np.nanmax(info["archive_flux"][observed]))
            top.set_title("Calibrated standard star vs. archival spectrum")
            top.grid(which="both", linestyle=":")
            top.minorticks_on()
            top.legend()
            calibrated_bands = info["observed_density"]*10**(0.4*sensitivity.magnitude(band))
            ratio = 100*(calibrated_bands/info["archive_flux"] - 1)
            bottom.plot(band[used], ratio[used], "o", color="C0", ms=4)
            bottom.plot(band[observed & ~used], ratio[observed & ~used], "x", color="gray", ms=6)
            bottom.axhline(0, color="k", linestyle="--")
            bottom.set_ylim(-residual_limit, residual_limit)
            bottom.set_ylabel("Calibrated/archival - 1 [%]")
            bottom.set_xlabel(r"Wavelength [$\AA$]")
            bottom.grid(which="both", linestyle=":")
            bottom.minorticks_on()
            plt.show()

        if std_star_dataset is not None:
            _, reference = self.list_available_standards(std_star_dataset=std_star_dataset)
            print("Please cite the reference of the archival spectrum:\n" + reference)
        self.sensitivity_function = sensitivity
        return sensitivity

    @_plot_style
    def target_flux_calibration(self, wavelengths_list, spec_atm_corrected_list, correction_factor, reddening = None, Rv = 3.1, wavelength_cuts = None,
                                output_directory = "./", save_spec = True, plot = True, file_format = "dat"):

        """
        This function calibrates the target spectra in flux density, using the sensitivity function obtained with the standard star:

            F_lambda = S(lambda) * counts / (exposure time * |d lambda/dx|),

        where S(lambda) is evaluated at the wavelengths of each target spectrum. Optionally, the spectra are also corrected for the Galactic
        reddening (Fitzpatrick 1999 extinction law).

        Parameters
        ----------
        wavelengths_list: list
            List with the wavelength solutions for all target spectra.
        spec_atm_corrected_list: list
            List with all the spectra corrected for atmospheric extinction.
        correction_factor: SensitivityFunction
            The flux calibration solution, i.e. the output of extraction.std_star_normalization().
        reddening: float
            Optional. The E(B-V) reddening in magnitudes. This is the Galactic reddening in the direction of your observation. You can look
            for the appropriate reddening for your target here: 'https://irsa.ipac.caltech.edu/applications/DUST/'.
        Rv: float
            Total to selective extinction ratio, A(V)/E(B-V). Default is 3.1 (the average value for the diffuse interstellar medium of the Milky Way).
        wavelength_cuts: list
            Optional. List with two wavelength values (in Angstroms), e.g. [3500,7500]. Only the spectrum within this range is kept. In any
            case, the spectra are restricted to the wavelength range where the sensitivity function is defined.
        output_directory: string
            A string with the path to the output directory.
        save_spec: boolean
            If True, the spectra are saved in the output directory as "TARGETNAME_spec_0.dat", "TARGETNAME_spec_1.dat" and so on.
        plot: boolean
            If True, easyspec plots the calibrated spectra with their uncertainties.
        file_format: string
            "dat" (default; text file, read by easyspec.analysis), "fits" (easyspec fits format, see easyspec.data_io.write_spectrum()) or
            "both". The columns are: wavelength (Angstrom), flux (erg/cm2/s/Angstrom), the systematic uncertainty of the wavelength solution
            (RMS of the fit, Angstrom), the statistical 1-sigma flux uncertainty, the 1-sigma uncertainty of the flux calibration (systematic,
            correlated between neighbouring wavelengths), and the mask of the extraction (0 = good).

        Returns
        -------
        wavelengths_sliced_list: list
            List with the wavelengths of the calibrated spectra (in increasing order).
        calibrated_flux_list: list
            List with the spectra calibrated in flux density (erg/cm2/s/Angstrom).

        Notes
        -----
        The statistical uncertainties, the calibration uncertainties and the masks of the calibrated spectra are stored in
        self.calibrated_flux_error_list, self.calibrated_flux_calibration_error_list and self.calibrated_mask_list.
        """

        if file_format not in ("dat", "fits", "both"):
            raise ValueError("'file_format' must be 'dat', 'fits' or 'both'.")
        if Rv is None:
            Rv = 3.1
        if self.exposure_target is None or not float(self.exposure_target) > 0:
            raise RuntimeError("The exposure time of the target is unknown. Please give it in extraction.import_data() (exposure_target, or the "
                               "header entry exposure_header_entry).")
        exposure = float(self.exposure_target)
        if len(wavelengths_list) != len(spec_atm_corrected_list):
            raise ValueError(f"'wavelengths_list' has {len(wavelengths_list)} wavelength solutions, but 'spec_atm_corrected_list' has "
                             f"{len(spec_atm_corrected_list)} spectra.")
        if isinstance(correction_factor, flux_calibration.SensitivityFunction):
            sensitivity = correction_factor
        else:
            raise TypeError("'correction_factor' must be the output of extraction.std_star_normalization() (a SensitivityFunction). The "
                            "correction arrays of previous easyspec versions are no longer supported: please run std_star_normalization() again.")
        if wavelength_cuts is not None and (len(wavelength_cuts) != 2 or wavelength_cuts[0] >= wavelength_cuts[1]):
            raise ValueError("'wavelength_cuts' must be [lambda_min, lambda_max], e.g. [3500, 7500].")

        def stored(name, number, shape):
            values = getattr(self, name, None)
            if values is not None and len(values) == len(spec_atm_corrected_list) and np.shape(values[number]) == shape:
                return np.asarray(values[number], dtype=float)
            return None

        dust_extinction_model = F99(Rv=Rv) if reddening is not None else None   # Fitzpatrick (1999) Milky Way R(V) dependent model
        output_directory = Path(output_directory)
        wavelengths_sliced_list, calibrated_flux_list, flux_error_list, calibration_error_list, mask_list = [], [], [], [], []
        for number, (wavelength, spectrum) in enumerate(zip(wavelengths_list, spec_atm_corrected_list)):
            wavelength = wavelength.to(u.AA).value if isinstance(wavelength, u.Quantity) else np.asarray(wavelength, dtype=float)
            spectrum = np.asarray(getattr(spectrum, "value", spectrum), dtype=float)
            if spectrum.shape != wavelength.shape:
                raise ValueError(f"Spectrum {number} has {spectrum.size} pixels, but its wavelength solution has {wavelength.size}.")

            # Variance: extinction-corrected (extinction_correction()), or from extracting(), or the uncertainties of previous versions.
            variance = stored("spec_atm_variance_list", number, spectrum.shape)
            if variance is None:
                variance = stored("spec_variance_list", number, spectrum.shape)
                if variance is not None and getattr(self, "extinction_factor_list", None) is not None and \
                   len(self.extinction_factor_list) == len(spec_atm_corrected_list):
                    variance = variance*np.asarray(self.extinction_factor_list[number])**2
            if variance is None:
                error = stored("spec_systematic_error_list", number, spectrum.shape)
                variance = None if error is None else error**2
            mask = stored("spec_mask_list", number, spectrum.shape)
            mask = np.zeros(spectrum.shape, dtype=np.uint8) if mask is None else mask.astype(np.uint8)

            dispersion = np.abs(np.gradient(wavelength))
            conversion = sensitivity(wavelength).value/(exposure*dispersion)
            flux = spectrum*conversion
            flux_error = np.sqrt(variance)*conversion if variance is not None else np.full(flux.shape, np.nan)
            calibration_error = np.abs(flux)*sensitivity.relative_error(wavelength)
            if dust_extinction_model is not None:
                try:
                    extinction = dust_extinction_model.extinguish(wavelength*u.AA, Ebv=reddening)
                except ValueError as error:
                    raise ValueError(f"The reddening correction could not be computed for spectrum {number} ({error}). The Fitzpatrick (1999) law is "
                                     "valid between ~1150 A and 3.3 micron.")
                flux, flux_error, calibration_error = flux/extinction, flux_error/extinction, calibration_error/extinction

            # Wavelength range: where the sensitivity function is defined, and within wavelength_cuts.
            keep = sensitivity.in_range(wavelength)
            requested = (wavelength.min(), wavelength.max()) if wavelength_cuts is None else \
                        (max(wavelength.min(), wavelength_cuts[0]), min(wavelength.max(), wavelength_cuts[1]))
            if sensitivity.wavelength_range[0] - requested[0] > 25 or requested[1] - sensitivity.wavelength_range[1] > 25:
                warnings.warn(f"Spectrum {number} covers {wavelength.min():.0f}-{wavelength.max():.0f} A, but the sensitivity function is defined only in "
                              f"{sensitivity.wavelength_range[0]:.0f}-{sensitivity.wavelength_range[1]:.0f} A: the spectrum was restricted to this range.")
            if wavelength_cuts is not None:
                keep &= (wavelength >= wavelength_cuts[0]) & (wavelength <= wavelength_cuts[1])
            if not np.any(keep):
                raise RuntimeError(f"Spectrum {number} has no pixels inside the calibrated range (and wavelength_cuts).")
            order = np.argsort(wavelength[keep])   # Increasing wavelengths
            wavelength, flux, flux_error, calibration_error, mask = (array[keep][order] for array in (wavelength, flux, flux_error, calibration_error, mask))

            wavelengths_sliced_list.append(wavelength*u.AA)
            calibrated_flux_list.append(flux*flux_calibration.FLAM_UNIT)
            flux_error_list.append(flux_error*flux_calibration.FLAM_UNIT)
            calibration_error_list.append(calibration_error*flux_calibration.FLAM_UNIT)
            mask_list.append(mask)

            if plot:
                plt.figure(figsize=(12,5))
                label = f"Spec {number}, E(B-V)={reddening}, R(V)={Rv}" if reddening is not None else f"Spec {number}, not corrected for reddening"
                plt.plot(wavelength, flux, color="orange", linewidth=0.8, label=label)
                if np.any(np.isfinite(flux_error)):
                    plt.fill_between(wavelength, flux - flux_error, flux + flux_error, color="C0", alpha=0.3, step="mid", label=r"$\pm1\sigma$ (statistical)")
                for low, high in flux_calibration.TELLURIC_BANDS:
                    if high > wavelength.min() and low < wavelength.max():
                        plt.axvspan(low, high, color="gray", alpha=0.12)
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                plt.xlim(wavelength.min(), wavelength.max())
                plt.ylim(min(0, np.nanpercentile(flux, 0.5)*1.2), np.nanpercentile(flux, 99.8)*1.3)
                plt.title(f"Calibrated spec {number} - "+str(self.target_name)+" field (gray: telluric bands, not corrected)")
                plt.ylabel(r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]", fontsize=12)
                plt.xlabel(r"Observed $\lambda$ [$\AA$]", fontsize=12)
                plt.legend()

            if save_spec:
                output_directory.mkdir(parents=True, exist_ok=True)
                wavelength_rms = 0.0
                if self.wavelengths_fit_rms_list is not None and len(self.wavelengths_fit_rms_list) == len(spec_atm_corrected_list):
                    wavelength_rms = float(getattr(self.wavelengths_fit_rms_list[number], "value", self.wavelengths_fit_rms_list[number]))
                base_name = output_directory / f"{self.target_name}_spec_{number}"
                if file_format in ("dat", "both"):
                    np.savetxt(f"{base_name}.dat", np.c_[wavelength, flux, wavelength_rms*np.ones(len(wavelength)), flux_error, calibration_error, mask],
                               fmt=["%.4f", "%.6e", "%.4f", "%.6e", "%.6e", "%d"],
                               header="wavelength (Angstrom), Flux (erg/cm2/s/Angstrom), Systematic wavelength error (Angstrom), Flux error "
                                      "(erg/cm2/s/Angstrom), Calibration flux error (erg/cm2/s/Angstrom), Mask ()")
                if file_format in ("fits", "both"):
                    header = {"OBJECT": str(self.target_name), "EXPTIME": exposure, "AIRMASS": float(self.airmass_target or np.nan),
                              "EBV": float(reddening) if reddening is not None else 0.0, "RV": float(Rv),
                              "STDFILE": Path(sensitivity.info.get("archive_file", "")).name}
                    data_io.write_spectrum(f"{base_name}.fits", wavelength*u.AA, flux*flux_calibration.FLAM_UNIT, flux_err=flux_error, mask=mask,
                                           header=header, extra_columns={"flux_cal_err": calibration_error*flux_calibration.FLAM_UNIT,
                                                                         "wavelength_sys_err": wavelength_rms*u.AA})

        if plot:
            plt.show()

        self.calibrated_flux_error_list = flux_error_list
        self.calibrated_flux_calibration_error_list = calibration_error_list
        self.calibrated_mask_list = mask_list
        return wavelengths_sliced_list, calibrated_flux_list

    @_plot_style
    def telluric_correction(self, wavelengths_list, calibrated_flux_list, spec_atm_corrected_std = None, wavelengths_std = None,
                            transmission_model = None, model_airmass = None, model_fwhm = None, bands = None, beta = 0.55, fit_bands = False,
                            continuum_window = 100, continuum_order = 1, output_directory = "./", save_spec = True, file_format = "dat", plots = True):

        """
        This function corrects the target spectra for the telluric absorption bands (O2 and H2O absorption in the Earth's atmosphere):
        the O2 B band (~6870 A) and A band (~7600 A), and the H2O bands at ~7200, ~8200 and ~9300 A.

        The telluric transmission T(lambda) can be obtained in two ways:

        1. From the standard star (give spec_atm_corrected_std and wavelengths_std): in each band, T is the ratio between the observed
           spectrum of the star and its continuum, interpolated from both sides of the band. This is the classical method (e.g. Wade & Horne
           1988, ApJ 324, 411; IRAF "telluric"). It works best with a hot star without intrinsic features in the bands, observed close in time
           and airmass to the target.
        2. From a transmission model computed by an external code, such as molecfit (Smette et al. 2015, A&A 576, A77; Kausch et al. 2015,
           A&A 576, A78), ESO SkyCalc or TAPAS (give transmission_model). The model is convolved to the resolution of the spectra.

        The transmission is scaled to the airmass of the target as T_target = T ** ((X_target/X_reference) ** beta), and the corrected
        spectrum is F/T_target. Optionally (fit_bands=True), the exponent and a small wavelength shift are fitted to each band of each target
        spectrum, minimizing the residual absorption (useful when the airmass scaling is not accurate, e.g. for the variable water bands).

        Parameters
        ----------
        wavelengths_list: list
            Wavelengths of the calibrated target spectra (output of extraction.target_flux_calibration()).
        calibrated_flux_list: list
            The calibrated target spectra (output of extraction.target_flux_calibration()).
        spec_atm_corrected_std: list or array
            Optional. The standard star spectrum corrected for atmospheric extinction (output of extraction.extinction_correction()), used to
            measure the telluric transmission.
        wavelengths_std: list or array
            Optional. The wavelength solution of the standard star (output of extraction.wavelength_calibration()).
        transmission_model: string, array or tuple
            Optional (instead of the standard star). A telluric transmission model: the path to a text file with two columns (wavelength,
            transmission) or to a fits table (e.g. the best-fit model of molecfit), an array with shape (N, 2), or a tuple (wavelength,
            transmission). Wavelengths in Angstrom (values below 100 are assumed to be in microns, as in molecfit).
        model_airmass: float
            Airmass for which the transmission model was computed. Default: the airmass of the target (no scaling).
        model_fwhm: float
            Spectral resolution (FWHM, in Angstrom) of the target spectra, used to convolve the transmission model. Default: estimated from the
            width of the lamp lines (if extraction.identify_lamp_lines() was used). Use 0 if the model already has the resolution of the
            spectra (e.g. molecfit fitted to these spectra).
        bands: dict
            Optional. The telluric bands to correct, as {name: [lambda_min, lambda_max]} in Angstrom. Default: telluric.TELLURIC_BANDS.
        beta: float
            Exponent of the airmass scaling. Default is 0.55, appropriate for the strong (saturated) bands at low/moderate resolution
            (Wade & Horne 1988); use 1 for weak, unsaturated absorption.
        fit_bands: boolean
            If True, the exponent and a wavelength shift (up to 1.5 A) of the transmission are fitted in each band of each target spectrum. The
            fit is used only if it improves the correction significantly; otherwise, the airmass scaling is kept. Requires spectra with a good
            signal-to-noise ratio in the bands. Default is False.
        continuum_window: float
            Width (Angstrom) of the regions on each side of a band used to interpolate the continuum. Default is 100 A.
        continuum_order: int
            Order of the polynomial continuum across each band. Default is 1.
        output_directory: string
            A string with the path to the output directory.
        save_spec: boolean
            If True, the corrected spectra are saved as "TARGETNAME_spec_0_telluric.dat" (or ".fits"), and so on, with the same columns as the
            files of target_flux_calibration().
        file_format: string
            "dat" (default), "fits" or "both".
        plots: boolean
            If True, easyspec plots the telluric transmission and the spectra before and after the correction.

        Returns
        -------
        wavelengths_list: list
            The wavelengths of the corrected spectra (the same as the input).
        corrected_flux_list: list
            The spectra corrected for the telluric absorption (erg/cm2/s/Angstrom).

        Notes
        -----
        The uncertainty of the transmission is propagated to the flux uncertainties. Pixels with strong absorption (transmission < 0.5) are
        flagged with bit 16 of the mask (corrected, but less reliable); pixels with transmission < 0.05 (nearly opaque) are set to np.nan and
        flagged with bit 4. The transmissions used are stored in self.telluric_transmission_list, the corrected uncertainties in
        self.telluric_flux_error_list, self.telluric_flux_calibration_error_list and self.telluric_mask_list, and the details (fitted
        exponents and shifts) in self.telluric_info.
        """

        if file_format not in ("dat", "fits", "both"):
            raise ValueError("'file_format' must be 'dat', 'fits' or 'both'.")
        if len(wavelengths_list) != len(calibrated_flux_list):
            raise ValueError(f"'wavelengths_list' has {len(wavelengths_list)} wavelength arrays, but 'calibrated_flux_list' has "
                             f"{len(calibrated_flux_list)} spectra.")
        use_standard = spec_atm_corrected_std is not None
        if use_standard == (transmission_model is not None):
            raise ValueError("Please give either the standard star spectrum (spec_atm_corrected_std and wavelengths_std) or a transmission_model.")
        bands = telluric.TELLURIC_BANDS if bands is None else {str(name): [min(b), max(b)] for name, b in bands.items()}
        if self.airmass_target is None:
            raise RuntimeError("The airmass of the target is unknown. Please give it in extraction.import_data().")
        airmass_target = float(self.airmass_target)

        def as_values(array):
            return array.to(u.AA).value if isinstance(array, u.Quantity) and array.unit.physical_type == "length" else \
                   np.asarray(getattr(array, "value", array), dtype=float)

        # ---------------------------------------------------------------- Reference transmission
        if use_standard:
            if wavelengths_std is None:
                raise ValueError("Please give also 'wavelengths_std', the wavelength solution of the standard star.")
            std_spectrum = spec_atm_corrected_std[0] if isinstance(spec_atm_corrected_std, (list, tuple)) else spec_atm_corrected_std
            std_wavelength = wavelengths_std[0] if isinstance(wavelengths_std, (list, tuple)) else wavelengths_std
            std_spectrum, std_wavelength = as_values(std_spectrum), as_values(std_wavelength)
            std_variance = None
            stored = getattr(self, "std_spec_atm_variance_list", None)
            if stored is not None and len(stored) == 1 and np.shape(stored[0]) == std_spectrum.shape:
                std_variance = np.asarray(stored[0], dtype=float)
            if self.airmass_std_star is None:
                raise RuntimeError("The airmass of the standard star is unknown. Please give it in extraction.import_data().")
            reference_airmass = float(self.airmass_std_star)
            order = np.argsort(std_wavelength)
            std_wavelength, std_spectrum = std_wavelength[order], std_spectrum[order]
            std_variance = std_variance[order] if std_variance is not None else None
            reference, reference_variance, used_bands = telluric.transmission_from_standard(std_wavelength, std_spectrum, std_variance, bands=bands,
                                                                                           window=continuum_window, order=continuum_order)
            reference_wavelength = std_wavelength
            source = f"standard star (airmass {reference_airmass:.3f})"
        else:
            # Transmission model: file (text or fits), array (N, 2) or tuple (wavelength, transmission)
            if isinstance(transmission_model, (str, Path)):
                model_path = Path(transmission_model)
                if model_path.suffix.lower() in (".fits", ".fit", ".fts"):
                    table = Table.read(model_path)
                    names = {name.lower(): name for name in table.colnames}
                    wave_name = next((names[n] for n in ("lambda", "wavelength", "wave", "lam") if n in names), None)
                    trans_name = next((names[n] for n in ("mtrans", "trans", "transmission", "trans_ma", "mtrans_ma") if n in names), None)
                    if wave_name is None or trans_name is None:
                        raise ValueError(f"Could not identify the wavelength and transmission columns of {model_path.name} (columns: "
                                         f"{table.colnames}). Please give the model as an array (wavelength, transmission).")
                    model_wavelength, model_transmission = np.asarray(table[wave_name], float), np.asarray(table[trans_name], float)
                else:
                    data = np.loadtxt(model_path, ndmin=2)
                    model_wavelength, model_transmission = data[:, 0], data[:, 1]
            elif isinstance(transmission_model, tuple) and len(transmission_model) == 2:
                model_wavelength, model_transmission = as_values(transmission_model[0]), as_values(transmission_model[1])
            else:
                data = np.asarray(transmission_model, dtype=float)
                if data.ndim != 2 or data.shape[1] < 2:
                    raise ValueError("'transmission_model' must be a file path, an array with shape (N, 2) or a tuple (wavelength, transmission).")
                model_wavelength, model_transmission = data[:, 0], data[:, 1]
            if np.nanmedian(model_wavelength) < 100:   # microns (e.g. molecfit)
                model_wavelength = model_wavelength*1e4
            if model_fwhm is None:
                info = getattr(self, "lamp_identification_info", None)
                if info:
                    fwhm_pixels = np.median([i["fwhm"] for i in info])
                    dispersion = np.median(np.abs(np.diff(as_values(wavelengths_list[0]))))
                    model_fwhm = float(fwhm_pixels*dispersion)
                    print(f"The transmission model is convolved to FWHM = {model_fwhm:.2f} A (from the width of the lamp lines).")
                else:
                    raise ValueError("Please give 'model_fwhm', the spectral resolution (FWHM, in Angstrom) of your spectra, to convolve the "
                                     "transmission model (or model_fwhm=0 if the model already has the resolution of the spectra).")
            reference_airmass = airmass_target if model_airmass is None else float(model_airmass)
            # The model is evaluated on a fine grid at the resolution of the spectra:
            fine_step = 0.25*max(model_fwhm, 4*np.median(np.diff(np.sort(model_wavelength))))/2.3548 if model_fwhm > 0 else \
                        np.median(np.diff(np.sort(model_wavelength)))
            reference_wavelength = np.arange(np.nanmin(model_wavelength), np.nanmax(model_wavelength), max(fine_step, 0.05))
            reference = telluric.model_to_spectrum(model_wavelength, model_transmission, reference_wavelength, model_fwhm)
            reference_variance = np.zeros(len(reference))
            used_bands = {name: band for name, band in bands.items() if band[1] > reference_wavelength[0] and band[0] < reference_wavelength[-1]}
            source = f"transmission model (airmass {reference_airmass:.3f}, FWHM = {model_fwhm:.2f} A)"

        if not used_bands:
            raise RuntimeError("None of the telluric bands could be measured in the reference (standard star or model). Check the wavelength "
                               "ranges, or give the bands with the parameter 'bands'.")

        def reference_transmission(wavelength):
            return np.interp(wavelength, reference_wavelength, reference, left=1.0, right=1.0)

        def reference_transmission_variance(wavelength):
            return np.interp(wavelength, reference_wavelength, reference_variance, left=0.0, right=0.0)

        # ---------------------------------------------------------------- Correction of each target spectrum
        airmass_ratio = airmass_target/reference_airmass
        power0 = airmass_ratio**beta
        print(f"Telluric correction from the {source}; target airmass = {airmass_target:.3f}; transmission scaled with exponent "
              f"(X_target/X_reference)^beta = {power0:.3f}. Bands: " + ", ".join(used_bands) + ".")

        corrected_list, error_list, calibration_error_list, mask_list, transmission_list, info_list = [], [], [], [], [], []
        for number, (wavelength, flux) in enumerate(zip(wavelengths_list, calibrated_flux_list)):
            wavelength_values, flux_values = as_values(wavelength), np.asarray(getattr(flux, "value", flux), dtype=float)
            if wavelength_values.shape != flux_values.shape:
                raise ValueError(f"Spectrum {number} has {flux_values.size} points, but its wavelength array has {wavelength_values.size}.")

            def stored(name):
                values = getattr(self, name, None)
                if values is not None and len(values) == len(calibrated_flux_list) and np.shape(values[number]) == flux_values.shape:
                    return np.asarray(getattr(values[number], "value", values[number]), dtype=float)
                return None
            flux_error = stored("calibrated_flux_error_list")
            calibration_error = stored("calibrated_flux_calibration_error_list")
            mask = stored("calibrated_mask_list")
            mask = np.zeros(flux_values.shape, dtype=np.uint8) if mask is None else mask.astype(np.uint8)
            variance = flux_error**2 if flux_error is not None else None

            transmission = np.ones(len(wavelength_values))
            transmission_variance = np.zeros(len(wavelength_values))
            band_info = {}
            for name, band in used_bands.items():
                inside = (wavelength_values >= band[0]) & (wavelength_values <= band[1])
                if np.count_nonzero(inside) < 3:
                    continue
                power, shift, improvement = power0, 0.0, 0.0
                if fit_bands:
                    others = [b for n, b in used_bands.items() if n != name]
                    fitted_power, fitted_shift, improvement = telluric.fit_band_scaling(wavelength_values, flux_values, variance,
                                                                                         reference_transmission, band, power0,
                                                                                         window=continuum_window, order=continuum_order,
                                                                                         other_bands=others)
                    if improvement > 25:   # Significant improvement (Delta chi2 > 25) with respect to the airmass scaling
                        power, shift = fitted_power, fitted_shift
                t_reference = np.clip(reference_transmission(wavelength_values[inside] - shift), 1e-4, None)
                transmission[inside] = t_reference**power
                # Relative uncertainty of T**power: power*sigma_T/T
                transmission_variance[inside] = (power*transmission[inside])**2*reference_transmission_variance(wavelength_values[inside] - shift)/t_reference**2
                band_info[name] = {"power": power, "shift": shift, "chi2_improvement": improvement,
                                   "min_transmission": float(np.min(transmission[inside]))}

            opaque = transmission < 0.05
            with np.errstate(divide="ignore", invalid="ignore"):
                corrected = flux_values/transmission
                corrected_error = np.sqrt((flux_error/transmission)**2 + corrected**2*transmission_variance/transmission**2) \
                                  if flux_error is not None else np.full(flux_values.shape, np.nan)
                corrected_calibration_error = calibration_error/transmission if calibration_error is not None else np.full(flux_values.shape, np.nan)
            mask = mask.copy()
            mask[transmission < 0.5] |= telluric.TELLURIC_MASK_BIT
            mask[opaque] |= 4
            corrected[opaque] = np.nan
            corrected_error[opaque] = np.nan

            corrected_list.append(corrected*flux_calibration.FLAM_UNIT)
            error_list.append(corrected_error*flux_calibration.FLAM_UNIT)
            calibration_error_list.append(corrected_calibration_error*flux_calibration.FLAM_UNIT)
            mask_list.append(mask)
            transmission_list.append(transmission)
            info_list.append({"source": source, "airmass_ratio": airmass_ratio, "beta": beta, "bands": band_info})
            fitted = [f"{n}: exponent {b['power']:.2f}, shift {b['shift']:+.1f} A" for n, b in band_info.items() if fit_bands]
            print(f"Spectrum {number}: corrected bands: " + (", ".join(band_info) if band_info else "none (no band in its wavelength range)")
                  + (" (" + "; ".join(fitted) + ")" if fitted else "") + ".")

            if plots and band_info:
                figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
                top.plot(wavelength_values, flux_values, color="C0", linewidth=0.7, alpha=0.8, label="Before the telluric correction")
                top.plot(wavelength_values, corrected, color="orange", linewidth=0.8, label="Telluric corrected")
                for name, band in used_bands.items():
                    for axis in (top, bottom):
                        axis.axvspan(band[0], band[1], color="gray", alpha=0.12)
                top.set_ylim(min(0, np.nanpercentile(corrected, 0.5)*1.2), np.nanpercentile(corrected, 99.8)*1.3)
                top.set_ylabel(r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]")
                top.set_title(f"Telluric correction - spec {number} - {self.target_name} field")
                top.minorticks_on()
                top.grid(which="both", linestyle=":")
                top.legend()
                bottom.plot(wavelength_values, transmission, color="k", linewidth=0.8)
                bottom.set_ylim(0, 1.1)
                bottom.set_ylabel("Transmission")
                bottom.set_xlabel(r"Observed $\lambda$ [$\AA$]")
                bottom.minorticks_on()
                bottom.grid(which="both", linestyle=":")

            if save_spec:
                output_directory_path = Path(output_directory)
                output_directory_path.mkdir(parents=True, exist_ok=True)
                wavelength_rms = 0.0
                if self.wavelengths_fit_rms_list is not None and len(self.wavelengths_fit_rms_list) == len(calibrated_flux_list):
                    wavelength_rms = float(getattr(self.wavelengths_fit_rms_list[number], "value", self.wavelengths_fit_rms_list[number]))
                base_name = output_directory_path / f"{self.target_name}_spec_{number}_telluric"
                if file_format in ("dat", "both"):
                    np.savetxt(f"{base_name}.dat", np.c_[wavelength_values, corrected, wavelength_rms*np.ones(len(wavelength_values)), corrected_error,
                                                        corrected_calibration_error, mask],
                               fmt=["%.4f", "%.6e", "%.4f", "%.6e", "%.6e", "%d"],
                               header="wavelength (Angstrom), Flux (erg/cm2/s/Angstrom), Systematic wavelength error (Angstrom), Flux error "
                                      "(erg/cm2/s/Angstrom), Calibration flux error (erg/cm2/s/Angstrom), Mask ()")
                if file_format in ("fits", "both"):
                    header = {"OBJECT": str(self.target_name), "AIRMASS": airmass_target, "TELLURIC": source[:60], "TELBETA": float(beta)}
                    data_io.write_spectrum(f"{base_name}.fits", wavelength_values*u.AA, corrected*flux_calibration.FLAM_UNIT, flux_err=corrected_error,
                                           mask=mask, header=header,
                                           extra_columns={"flux_cal_err": corrected_calibration_error*flux_calibration.FLAM_UNIT,
                                                          "wavelength_sys_err": wavelength_rms*u.AA, "telluric_transmission": transmission})

        if plots:
            plt.figure(figsize=(12, 4))
            shown = np.zeros(len(reference_wavelength), dtype=bool)
            for band in used_bands.values():
                shown |= (reference_wavelength >= band[0] - continuum_window) & (reference_wavelength <= band[1] + continuum_window)
            plt.plot(reference_wavelength[shown], np.where(shown, reference, np.nan)[shown], ".", ms=2, color="k", label=f"Reference: {source}")
            for name, band in used_bands.items():
                plt.axvspan(band[0], band[1], color="gray", alpha=0.12)
                plt.text(0.5*(band[0] + band[1]), 1.12, name, ha="center", fontsize=8)
            plt.ylim(0, 1.2)
            plt.xlabel(r"Wavelength [$\AA$]")
            plt.ylabel("Transmission")
            plt.title("Telluric transmission (before the airmass scaling)")
            plt.minorticks_on()
            plt.grid(which="both", linestyle=":")
            plt.legend(loc="lower left")
            plt.show()

        self.telluric_transmission_list = transmission_list
        self.telluric_flux_error_list = error_list
        self.telluric_flux_calibration_error_list = calibration_error_list
        self.telluric_mask_list = mask_list
        self.telluric_info = info_list
        return list(wavelengths_list), corrected_list
