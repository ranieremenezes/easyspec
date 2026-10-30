import numpy as np
import matplotlib.pyplot as plt
import corner
from pathlib import Path
import types
import re
import warnings
import io
import contextlib
from functools import wraps
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import trapezoid
from astropy import units as u
from astropy.table import Table
from easyspec import data_io
from .line_models import model_Gauss, model_Lorentz, model_Voigt, model_skewed_gaussian, model_skewed_lorentzian
from .aux_fit_lines import (aux_validate_line_names, aux_ensure_numpy_array, aux_validate_models, aux_generate_line_names, aux_estimate_fwhm,
                            aux_group_blended_lines, aux_automatic_priors, aux_parse_custom_priors, aux_run_mcmc, aux_summarize_posterior,
                            ln_prior, ln_likelihood, ln_probability, MODEL_PARAMETERS)
from . import continuum_detection
from . import aux_line_physics
from . import aux_stacking
from . import aux_identify_lines
from easyspec.extraction import line_identification

libpath = Path(__file__).parent.resolve() / Path("lines")
FLAM_UNIT = u.erg/u.cm**2/u.s/u.AA

# The functions below use extraction.find_nearest(); a lightweight replacement avoids creating an instance of the extraction class (and its
# side effects) when easyspec.analysis is imported.
extraction = types.SimpleNamespace(find_nearest=continuum_detection.find_nearest)

# Bits of the 1D data quality mask (see easyspec.extraction) that indicate unreliable pixels:
BAD_SPECTRUM_BITS = 1 | 4 | 8


def _plot_style(function):

    """Apply the easyspec plotting style (font size) only inside the decorated method, without changing the global settings of the user."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context({"font.size": 12}):
            return function(*args, **kwargs)
    return wrapper


class CombinedModel:
    """A picklable class that acts like a function representing one line model or combinations of line models"""

    def __init__(self, model_names):
        self.model_names = model_names
        self.param_counts = {
            'Gaussian': 3,          # peak location, amplitude, FWHM
            'Lorentz': 3,           # peak location, amplitude, FWHM
            'Voigt': 4,             # peak location, amplitude, FWHM of the Gaussian, FWHM of the Lorentzian
            'Skewedgaussian': 4,    # peak location, amplitude, FWHM, skewness
            'Skewedlorentzian': 4   # peak location, amplitude, FWHM, skewness
        }
        unknown = [name for name in model_names if name not in self.param_counts]
        if unknown:
            raise ValueError(f"Unknown line model(s): {unknown}. Options are {list(self.param_counts)}.")

    def __call__(self, theta, x):
        """Make the class callable like a function. This is important for emcee"""
        final_model = 0
        param_index = 0

        for name in self.model_names:
            num_params = self.param_counts[name]
            comp_theta = theta[param_index:param_index + num_params]
            param_index += num_params

            if name == "Gaussian":
                final_model += model_Gauss(comp_theta, x)
            elif name == "Lorentz":
                final_model += model_Lorentz(comp_theta, x)
            elif name == "Voigt":
                final_model += model_Voigt(comp_theta, x)
            elif name == "Skewedgaussian":
                final_model += model_skewed_gaussian(comp_theta, x)
            elif name == "Skewedlorentzian":
                final_model += model_skewed_lorentzian(comp_theta, x)

        return final_model


class analysis:

    """This class contains all the functions necessary to perform the analysis of calibrated spectral data."""

    def __init__(self):

        self.line_models = types.SimpleNamespace()
        self.line_models.model_Gauss = model_Gauss
        self.line_models.model_Lorentz = model_Lorentz
        self.line_models.model_Voigt = model_Voigt
        self.line_models.model_skewed_gaussian = model_skewed_gaussian
        self.line_models.model_skewed_lorentzian = model_skewed_lorentzian

        # Filled by load_calibrated_data():
        self.target_name = None
        self.output_dir = "."
        self.wavelengths = None
        self.flux_err = None
        self.flux_calibration_error = None
        self.spectrum_mask = None
        self.wavelength_systematic_error = None
        self.flux_systematic_error = None
        self.instrumental_fwhm = None
        # Filled by find_lines():
        self.continuum_regions = None
        self.continuum_mask = None
        self.continuum_noise = None

    # ==================================================================================================================================
    # Helpers
    # ==================================================================================================================================

    @staticmethod
    def _values(array):
        """Plain numpy array (without units)."""
        return np.asarray(getattr(array, "value", array), dtype=float)

    def _matching(self, array, n_pixels):
        """The stored array (flux_err, mask, ...) if it corresponds to a spectrum with n_pixels pixels, else None."""
        if array is None:
            return None
        values = np.asarray(getattr(array, "value", array))
        return values if values.shape == (n_pixels,) else None

    def _good_pixels(self, wavelengths, flux):
        """Pixels with finite data and not flagged as bad in the data quality mask of the loaded spectrum."""
        good = np.isfinite(wavelengths) & np.isfinite(flux)
        mask = self._matching(self.spectrum_mask, len(flux))
        if mask is not None:
            good &= (mask.astype(int) & BAD_SPECTRUM_BITS) == 0
        return good

    # ==================================================================================================================================
    # Continuum
    # ==================================================================================================================================

    def find_continuum_regions(self, wavelengths, flux_density, flux_err = None, knot_spacing = None, detection_sigma = 3.0,
                               extension_sigma = 1.0, min_region_pixels = 5, plot = True):

        """
        This function finds automatically the continuum (line-free) regions of a spectrum, for any kind of object: AGN (with broad emission
        lines), stars (with absorption lines), galaxies (with narrow emission lines) and so on.

        How it works: a smooth curve (a cubic spline with widely spaced knots, which cannot follow the lines) is fitted to the spectrum. The
        residuals are smoothed on several scales, from a few pixels (narrow lines) up to a fraction of the knot spacing (broad lines), and the
        lines are detected with two thresholds: a line is seeded where the residual exceeds 'detection_sigma' and extended (with the same sign)
        while it exceeds 'extension_sigma', which captures the faint wings of broad lines. The lines are masked and the curve is fitted again,
        until the mask does not change. The continuum regions are the remaining (line-free) intervals.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        flux_err: numpy.ndarray
            Optional. The 1-sigma uncertainty of each pixel. Default: the uncertainty read by load_calibrated_data() (if it corresponds to
            this spectrum); otherwise, the noise is estimated from the data.
        knot_spacing: float
            Optional. Distance (in Angstroms) between the knots of the smooth curve. It must be larger than the full width of the broadest lines
            (including their wings), but small enough to follow the shape of the continuum. Default: 1/6 of the wavelength range. Increase it
            if the wings of very broad lines are taken as continuum; decrease it if a curved continuum is taken as a line.
        detection_sigma: float
            Detection threshold of the lines, in units of the noise of the smoothed residuals. Default is 3.
        extension_sigma: float
            Threshold used to extend the lines (their wings). Default is 1.
        min_region_pixels: int
            Minimum number of pixels of a continuum region. Default is 5.
        plot: boolean
            If True, the spectrum is shown with the continuum regions.

        Returns
        -------
        continuum_regions: list
            List with the continuum regions, e.g. [[3300, 3650], [3720, 4200], ...], in Angstroms. It can be edited and passed to
            find_lines(continuum_regions=...).
        """

        x, f = self._values(wavelengths), self._values(flux_density)
        good = self._good_pixels(x, f)
        if flux_err is None:
            flux_err = self._matching(self.flux_err, len(f))
        flux_err = None if flux_err is None else self._values(flux_err)
        continuum_mask, smooth, _ = continuum_detection.find_continuum_mask(x, f, flux_err=flux_err, good=good, knot_spacing=knot_spacing,
                                                                             detection_sigma=detection_sigma, extension_sigma=extension_sigma)
        regions = continuum_detection.mask_to_regions(x, continuum_mask, min_pixels=min_region_pixels)
        fraction = np.count_nonzero(continuum_mask)/max(1, np.count_nonzero(good))
        print(f"{len(regions)} continuum regions found, covering {100*fraction:.0f}% of the spectrum.")
        if fraction < 0.2:
            warnings.warn("Less than 20% of the spectrum was identified as continuum. This is expected for spectra dominated by lines or by "
                          "molecular bands (e.g. late-type stars), where the 'continuum' is a pseudo-continuum. Check the plot, and consider "
                          "giving the continuum regions by hand.")

        if plot:
            with plt.rc_context({"font.size": 12}):
                plt.figure(figsize=(12, 4.5))
                plt.plot(x, f, color="orange", linewidth=0.8, label="Spectrum")
                plt.plot(x, smooth, color="C0", linewidth=1, label="Smooth continuum (auxiliary)")
                for number, region in enumerate(regions):
                    plt.axvspan(region[0], region[1], color="C2", alpha=0.15, label="Continuum regions" if number == 0 else None)
                ylim = np.nanpercentile(f[good], [0.5, 99.5])
                plt.ylim(min(0, ylim[0]), 1.3*ylim[1])
                plt.xlim(np.nanmin(x), np.nanmax(x))
                plt.xlabel(r"Observed $\lambda$ [$\AA$]")
                plt.ylabel(r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]")
                plt.title(f"Automatic continuum regions - {self.target_name}")
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                plt.legend()
                plt.show()

        self.continuum_regions = regions
        return regions

    def continuum_fit(self, flux, wavelengths, continuum_regions, method = "powerlaw", pl_order = 2, smooth_window = 111, flux_err = None,
                      knot_spacing = None, good = None, noise_window = 101):

        """
        This function fits the continuum emission within the continuum regions.

        Parameters
        ----------
        flux: array (float)
            Array with the flux density.
        wavelengths: array (float)
            Wavelengths, in absolute values (i.e. without units).
        continuum_regions: list, "auto" or None
            The continuum is fitted only within these wavelength regions and then interpolated/extrapolated everywhere else, e.g.
            continuum_regions = [[3000,6830],[6930,7500]]. With "auto", the regions are found automatically (see find_continuum_regions()).
            If None, the whole spectrum is used (only recommended if the lines are weak).
        method: string
            "powerlaw" (or "pl", "polynomial"): a polynomial of order pl_order in wavelength (despite the historical name, it is a polynomial),
            fitted with weights and sigma clipping. Better for spectra with broad lines.
            "median_filter": the continuum pixels are interpolated across the lines and smoothed with a median filter of smooth_window pixels.
            Excellent for spectra with narrow lines.
            "spline": a cubic spline with knots every knot_spacing Angstroms (default: 1/8 of the range); flexible, for curved continua.
        pl_order: int
            Order of the polynomial (method = "powerlaw").
        smooth_window: int
            Size (in pixels) of the median filter (method = "median_filter"). It must be odd.
        flux_err: array
            Optional. Uncertainty of each pixel, used to weight the fit.
        knot_spacing: float
            Optional. Knot spacing (in Angstroms) for method = "spline" (and for the automatic regions).
        good: array of bool
            Optional. Pixels that can be used.
        noise_window: int
            Size (in pixels) of the running window used to measure the noise of the continuum at each wavelength (stored in
            self.continuum_noise). Default is 101.

        Returns
        -------
        continuum_selection: array (float)
            Array with the continuum flux density at each wavelength.
        continuum_std_deviation: list
            The (robust) standard deviation of the residuals in each continuum region.
        """

        x, f = self._values(wavelengths), self._values(flux)
        good = np.isfinite(x) & np.isfinite(f) if good is None else np.asarray(good, dtype=bool) & np.isfinite(f)
        if isinstance(continuum_regions, str):
            if continuum_regions.lower() != "auto":
                raise ValueError("'continuum_regions' must be a list of [lambda_min, lambda_max] regions, 'auto' or None.")
            mask, _, _ = continuum_detection.find_continuum_mask(x, f, flux_err=flux_err, good=good, knot_spacing=knot_spacing)
            continuum_regions = continuum_detection.mask_to_regions(x, mask)
        elif continuum_regions is None:
            continuum_regions = [[float(np.nanmin(x)), float(np.nanmax(x))]]
        elif np.ndim(continuum_regions[0]) == 0:
            continuum_regions = [list(continuum_regions)]
        continuum_regions = sorted([[min(r), max(r)] for r in continuum_regions])

        continuum_mask = continuum_detection.regions_to_mask(x, continuum_regions) & good
        if np.count_nonzero(continuum_mask) < 5:
            raise ValueError(f"The continuum regions {continuum_regions} contain fewer than 5 valid pixels of the spectrum "
                             f"({np.nanmin(x):.0f}-{np.nanmax(x):.0f} A).")
        weights = None
        if flux_err is not None:
            flux_err = self._values(flux_err)
            weights = np.where(np.isfinite(flux_err) & (flux_err > 0), 1/np.where(flux_err > 0, flux_err, 1.0), 0.0)
        continuum_selection = continuum_detection.fit_continuum(x, f, continuum_mask, method=method, pl_order=pl_order,
                                                                smooth_window=smooth_window, knot_spacing=knot_spacing, weights=weights)
        noise_per_region, _ = continuum_detection.region_noise(x, f - continuum_selection, continuum_mask, continuum_regions)
        # Noise at each wavelength, from the continuum residuals in a running window (more local than the regions themselves):
        noise = continuum_detection.local_noise(x, f - continuum_selection, continuum_mask, window=noise_window)
        self.continuum_regions = continuum_regions
        self.continuum_mask = continuum_mask
        self.continuum_noise = noise
        return continuum_selection, list(noise_per_region)

    # ==================================================================================================================================
    # Data
    # ==================================================================================================================================

    @_plot_style
    def load_calibrated_data(self, calibrated_spec_data, target_name = None, output_dir = "./", plot = True):

        """
        This function loads a calibrated spectrum: the files 'TARGETNAME_spec_X.dat' (or '.fits') generated by extraction.target_flux_calibration()
        or extraction.telluric_correction(), or the text files of previous easyspec versions.

        Parameters
        ----------
        calibrated_spec_data: string
            The path to the spectrum file ('.dat' or '.fits').
        target_name: string
            Optional. This name will be used in all subsequent plots and output files. Default: the name of the file.
        output_dir: string
            A string with the path to the output directory (it is created if it does not exist).
        plot: boolean
            If True, the spectrum will be shown (with its 1-sigma uncertainty, if available).

        Returns
        -------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.

        Notes
        -----
        The other columns of the file are stored as attributes: self.flux_err (statistical 1-sigma uncertainty of the flux),
        self.flux_calibration_error (uncertainty of the flux calibration), self.spectrum_mask (data quality mask, 0 = good) and
        self.wavelength_systematic_error (accuracy of the wavelength solution) and self.instrumental_fwhm (spectral resolution, FWHM in
        Angstrom at each wavelength, used by line_physics() to correct the line widths). They are None if the file does not have them.
        """

        path = Path(calibrated_spec_data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        self.output_dir = str(Path(output_dir))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.target_name = path.stem if target_name is None else str(target_name)

        table = data_io.read_spectrum(path)

        def column(*names):
            for name in names:
                if name in table.colnames:
                    return table[name]
            return None

        wavelengths = u.Quantity(table["wavelength"])
        if wavelengths.unit == u.dimensionless_unscaled:
            wavelengths = wavelengths.value*u.AA
        flux_density = u.Quantity(table["flux"])
        if flux_density.unit == u.dimensionless_unscaled:
            flux_density = flux_density.value*FLAM_UNIT

        # Columns written by the different easyspec versions:
        flux_err = column("flux_err", "flux_error", "flux_sys_err", "col3")
        calibration_error = column("flux_cal_err", "calibration_flux_error", "col4")
        mask = column("mask", "col5")
        wavelength_error = column("wavelength_sys_err", "systematic_wavelength_error", "col2")
        instrumental = column("instrumental_fwhm")

        def as_flux(values):
            if values is None:
                return None
            values = u.Quantity(values)
            return values.value*flux_density.unit if values.unit == u.dimensionless_unscaled else values.to(flux_density.unit)

        order = np.argsort(wavelengths.value)   # Increasing wavelengths
        wavelengths, flux_density = wavelengths[order], flux_density[order]
        self.flux_err = as_flux(flux_err)[order] if flux_err is not None else None
        self.flux_calibration_error = as_flux(calibration_error)[order] if calibration_error is not None else None
        self.spectrum_mask = np.asarray(getattr(mask, "value", mask)).astype(int)[order] if mask is not None else None
        if wavelength_error is not None:
            wavelength_error = u.Quantity(wavelength_error)
            value = float(np.nanmedian(wavelength_error.value))
            self.wavelength_systematic_error = value*(u.AA if wavelength_error.unit == u.dimensionless_unscaled else wavelength_error.unit)
        else:
            self.wavelength_systematic_error = None
        self.instrumental_fwhm = None
        if instrumental is not None:
            values = np.asarray(getattr(instrumental, "value", instrumental), dtype=float)[order]
            if np.any(np.isfinite(values) & (values > 0)):
                self.instrumental_fwhm = values*u.AA
        # Kept for compatibility with the functions that save the fit results:
        self.flux_systematic_error = self.flux_err

        n_bad = 0 if self.spectrum_mask is None else int(np.count_nonzero(self.spectrum_mask & BAD_SPECTRUM_BITS))
        print(f"{self.target_name}: {len(wavelengths)} pixels, {wavelengths.value.min():.1f}-{wavelengths.value.max():.1f} A"
              + ("; with flux uncertainties" if self.flux_err is not None else "; no flux uncertainties in the file")
              + (f"; {n_bad} pixel(s) flagged as bad in the mask" if n_bad else "")
              + (f"; instrumental resolution FWHM = {np.nanmedian(self.instrumental_fwhm.value):.2f} A" if self.instrumental_fwhm is not None else "")
              + ".")

        if plot:
            x, f = wavelengths.value, flux_density.value
            plt.figure(figsize=(12,5))
            if self.flux_err is not None:
                e = self.flux_err.value
                plt.fill_between(x, f - e, f + e, color="C0", alpha=0.3, step="mid", label=r"$\pm1\sigma$")
            plt.plot(x, f, color='orange', linewidth=0.8, label=self.target_name+" - calibrated data")
            if n_bad:
                bad = (self.spectrum_mask & BAD_SPECTRUM_BITS) > 0
                plt.plot(x[bad], np.nan_to_num(f[bad]), "x", color="red", ms=4, label="Flagged pixels")
            finite = np.isfinite(f)
            ylim = np.nanpercentile(f[finite], [0.5, 99.8])
            plt.ylim(min(0, 1.2*ylim[0]), 1.2*ylim[1])
            plt.xlim(x.min(), x.max())
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.title(f"Calibrated data - {self.target_name}")
            plt.legend()
            plt.ylabel(r"F$_{\lambda}$ "+f"[{flux_density.unit}]",fontsize=12)
            plt.xlabel(r"Observed $\lambda$ [$\AA$]",fontsize=12)
            plt.show()

        self.wavelengths = wavelengths
        return wavelengths, flux_density

    # ==================================================================================================================================
    # Line detection
    # ==================================================================================================================================


    @_plot_style
    def find_lines(self, wavelengths, flux_density, line_significance_sigma = 5, peak_distance = 4, peak_width = 2, method = "median_filter",
                   continuum_regions = None, pl_order = 2, smooth_window = 111, plot_lines = True, plot_regions = True, save_plot = False,
                   knot_spacing = None, noise = "continuum"):

        """
        This function finds all emission and absorption lines with significance above 'line_significance_sigma' with respect to the local
        continuum.

        The continuum is fitted within the continuum regions (given by hand, or found automatically with continuum_regions="auto"), and its
        noise is measured in each region and interpolated at all wavelengths. The lines are then searched on the whole continuum-subtracted
        spectrum at once, so a line is detected wherever it is, also between two continuum regions.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        line_significance_sigma: float
            Defines how many standard deviations (see 'noise') the line peak must be above (or below) the continuum to be detected.
        peak_distance: float
            The minimal distance (in pixels, not in Angstroms) between peaks (>= 1). Default is 4, about one resolution element of a typical
            long-slit spectrum, so that close lines (e.g. Ca II H and K, or [NII] next to H-alpha) are detected separately. Increase it to
            avoid several detections in a very broad line.
        peak_width: float
            Minimum required width of the peaks, in pixels (measured at half of their prominence). Default is 2, which rejects single-pixel
            spikes but keeps unresolved lines.
        method: string
            The method used to fit the continuum: "median_filter" (default; excellent for narrow lines), "powerlaw" (a polynomial of order
            pl_order; better for broad lines) or "spline". See continuum_fit().
        continuum_regions: list, "auto" or None
            The wavelength regions (in Angstroms) used to fit the continuum and to measure its noise, e.g. [[3000,6830],[6930,7500]].
            With "auto", easyspec finds the continuum regions automatically (see find_continuum_regions()); they are stored in
            self.continuum_regions, so you can check and edit them. If None, the whole spectrum is used (only recommended if the lines are weak).
        pl_order: int
            Polynomial order for method = "powerlaw".
        smooth_window: int
            Size of the median filter (in pixels) for method = "median_filter". It must be odd.
        plot_lines: boolean
            If True, the spectrum, the continuum and all detected lines will be shown.
        plot_regions: boolean
            If True, the continuum-subtracted spectrum is shown with the continuum regions and the detection threshold.
        save_plot: boolean
            If True, the spectrum plot will be saved in the output directory defined in analysis.load_calibrated_data().
        knot_spacing: float
            Optional. Knot spacing (in Angstroms) for continuum_regions = "auto" and for method = "spline".
        noise: string
            The noise used for the detection threshold:
            - "continuum" (default): the scatter of the spectrum around the continuum, measured in the continuum regions. It includes any
              real structure (weak features, absorption forests, calibration ripples), so only lines that stand out from this structure are
              detected. Recommended for stars with many absorption lines.
            - "statistical": the 1-sigma uncertainty of each pixel (flux_err column of the spectrum file). It detects weaker lines (e.g. NV
              on the wing of Ly-alpha, or HeII in quasars), but in high signal-to-noise spectra of cool stars it detects the whole absorption
              forest. Requires a spectrum file with flux uncertainties.

        Returns
        -------
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum flux density. Standard easyspec units are in erg/cm2/s/A.
        line_std_deviation: numpy.ndarray (float)
            The standard deviation of the local continuum at each line. The line significance is calculated with respect to this value.
        wavelength_peak_positions: numpy.ndarray (astropy.units Angstrom)
            The position of each peak in Angstroms.
        peak_heights: numpy.ndarray (astropy.units erg/cm2/s/A)
            The flux density at each peak, in erg/cm2/s/A.
        line_significance: array (floats)
            The line significance with respect to the local continuum standard deviation (negative for absorption lines).

        Notes
        -----
        All outputs are sorted by wavelength. The noise of the continuum at each wavelength is stored in self.continuum_noise, and the type
        of each line ("emission" or "absorption") in self.line_types.
        """

        x, f = self._values(wavelengths), self._values(flux_density)
        flux_unit = getattr(flux_density, "unit", FLAM_UNIT)
        good = self._good_pixels(x, f)
        flux_err = self._matching(self.flux_err, len(f))

        if isinstance(continuum_regions, str) and continuum_regions.lower() == "auto":
            continuum_regions = self.find_continuum_regions(wavelengths, flux_density, flux_err=flux_err, knot_spacing=knot_spacing, plot=False)

        continuum_baseline, _ = self.continuum_fit(f, x, continuum_regions, method=method, pl_order=pl_order, smooth_window=smooth_window,
                                                   flux_err=flux_err, knot_spacing=knot_spacing, good=good)
        residual = f - continuum_baseline
        if str(noise).lower() == "statistical":
            if flux_err is None:
                raise ValueError("noise='statistical' requires the flux uncertainties of the spectrum (the flux_err column of the files of "
                                 "extraction.target_flux_calibration()). Use noise='continuum' instead.")
            noise = np.where(np.isfinite(flux_err) & (flux_err > 0), flux_err, np.nanmedian(flux_err))
        elif str(noise).lower() == "continuum":
            noise = self.continuum_noise
        else:
            raise ValueError("'noise' must be 'continuum' or 'statistical'.")

        index, kinds = continuum_detection.detect_lines(x, residual, noise, threshold_sigma=line_significance_sigma, peak_distance=peak_distance,
                                                        peak_width=peak_width, good=good)
        if len(index) == 0:
            raise RuntimeError("No significant emission or absorption line was found. You can try a lower 'line_significance_sigma' or 'peak_width', "
                               "or check the continuum regions (plot_regions=True).")

        peak_heights = f[index]*flux_unit
        wavelength_peak_positions = x[index]*u.AA
        line_std_deviation = noise[index]
        line_significance = residual[index]/noise[index]
        self.line_types = list(kinds)
        print(f"{len(index)} line(s) found: " + ", ".join(f"{w:.1f} A ({'em' if k == 'emission' else 'abs'}, {s:.0f} sigma)"
                                                       for w, k, s in zip(x[index], kinds, np.abs(line_significance))) + ".")

        regions = self.continuum_regions
        if plot_regions:
            plt.figure(figsize=(12,4))
            plt.plot(x, residual, color="k", linewidth=0.6, label="Continuum-subtracted spectrum")
            plt.fill_between(x, -line_significance_sigma*noise, line_significance_sigma*noise, color="C1", alpha=0.3,
                             label=fr"$\pm${line_significance_sigma}$\sigma$ ({'continuum' if np.ndim(noise) and noise is self.continuum_noise else 'statistical'} noise)")
            for number, region in enumerate(regions):
                plt.axvspan(region[0], region[1], color="C2", alpha=0.15, label="Continuum regions" if number == 0 else None)
            plt.plot(x[index], residual[index], "v", color="C3", ms=6, label="Detected lines")
            ylim = np.nanpercentile(residual[good], [0.2, 99.8])
            noise_level = line_significance_sigma*np.nanpercentile(noise[good], 95)
            plt.ylim(1.3*min(ylim[0], -noise_level), 1.3*max(ylim[1], noise_level))
            plt.xlim(x.min(), x.max())
            plt.ylabel(r"F$_{\lambda}$ - continuum "+f"[{flux_unit}]", fontsize=12)
            plt.xlabel(r"Observed $\lambda$ [$\AA$]", fontsize=12)
            plt.title("Continuum regions (green) and detection threshold (orange)")
            plt.minorticks_on()
            plt.grid(which="both", linestyle=":")
            plt.legend(fontsize=9, loc="upper right")

        if plot_lines:
            plt.figure(figsize=(12,5))
            top = np.nanpercentile(f[good], 99.8)
            for w, height, kind, baseline in zip(x[index], f[index], kinds, continuum_baseline[index]):
                if kind == "emission":
                    plt.text(w, height + 0.05*top, f"{w:.1f}"+r"$\AA$", color="C0", rotation=90, fontsize=10, ha="center", va="bottom")
                else:
                    plt.vlines(w, height, baseline + 0.15*top, color="red", linewidth=0.8)
                    plt.text(w, baseline + 0.17*top, f"{w:.1f}"+r"$\AA$", color="red", rotation=90, fontsize=10, ha="center", va="bottom")
            method_label = {"median_filter": "Median-filter", "spline": "Spline"}.get(str(method).lower(), "Polynomial")
            plt.plot(x, continuum_baseline, label=f"{method_label} continuum")
            plt.plot(x, f, color='orange', linewidth=0.8)
            plt.plot(x, residual, alpha=0.15, color='black', label="Continuum-subtracted spec")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(x.min(), x.max())
            plt.ylim(min(0, 1.1*np.nanmin(residual[good])), 1.5*top)
            plt.ylabel(r"F$_{\lambda}$ "+f"[{flux_unit}]",fontsize=12)
            plt.xlabel(r"Observed $\lambda$ [$\AA$]",fontsize=12)
            plt.title(str(self.target_name))
            plt.legend()
            plt.tight_layout()
            if save_plot:
                plt.savefig(Path(self.output_dir) / f"{self.target_name}_line_wavelengths.pdf", bbox_inches='tight')

        if plot_lines or plot_regions:
            plt.show()

        return continuum_baseline, line_std_deviation, wavelength_peak_positions, peak_heights, line_significance
  
    @_plot_style
    def identify_lines(self, wavelength_peak_positions, line_significance = None, wavelengths = None, flux_density = None, redshift = None,
                       redshift_range = (0, 7), tolerance = 1500, line_types = None, use_tentative = False, plot = True):

        """
        This function identifies the lines found by find_lines() and estimates the redshift of the source, automatically.

        The redshift is found by pattern matching with a list of strong reference lines (e.g. Ly-alpha, C IV, C III], Mg II, [O II], H-beta,
        [O III], H-alpha, and the Ca II H&K, Mg b and Na D absorption lines): every pair (detected line, reference line) gives a candidate
        redshift, and each candidate is scored by how many detected lines it explains, weighted by the typical strength of the reference
        lines and by the significance of the detections, with penalties for strong detections left unexplained and for strong emission lines
        that should be in the spectrum but were not detected. All lines are then named at the best redshift: with the reference lines, or
        (tentatively) with the easyspec line database (lines/astro_lines.dat). Absorption lines inside telluric bands, and pairs of absorption
        lines matching the doublets of intervening absorbers (Mg II, C IV, Si IV, Ca II), are flagged.

        Parameters
        ----------
        wavelength_peak_positions: numpy.ndarray (astropy.units Angstrom)
            The observed wavelengths of the lines (output of analysis.find_lines()).
        line_significance: numpy.ndarray
            Optional. The significance of each line (output of analysis.find_lines(); negative for absorption lines). Stronger lines weigh
            more in the redshift search. If None, all lines weigh the same.
        wavelengths, flux_density: numpy.ndarray (astropy.units)
            Optional. The spectrum, used for the plot and to set the minimum tolerance (2 pixels).
        redshift: float
            Optional. If you know the redshift, the lines are identified at this redshift (no search).
        redshift_range: [float, float]
            The redshift interval searched. Default is (0, 7).
        tolerance: float
            Maximum velocity difference (km/s) between a detected line and the expected position of a reference line. Default is 1500 km/s,
            suitable for the broad lines of quasars (whose high-ionization lines are often shifted by ~1000 km/s); about 300 km/s is enough for
            the narrow lines of galaxies.
        line_types: list
            Optional. "emission" or "absorption" for each line. Default: from the sign of line_significance or, if this is not given, from
            find_lines() (self.line_types).
        use_tentative: bool
            If True, the lines identified only tentatively (with the line database, not with the reference lines) are also returned.
        plot: bool
            If True (and the spectrum is given), the spectrum is shown with the identified lines and the expected positions of other strong lines.

        Returns
        -------
        redshift: float
            The estimated redshift (a first guess: fit_lines() measures the redshift of each line accurately). NaN if no solution was found.
        rest_frame_line_wavelengths: list
            The rest-frame wavelengths of the identified lines (easyspec convention: vacuum below 2000 A, air above), ready for fit_lines().
        line_names: list
            Their names.
        line_index: numpy.ndarray (int)
            The indices of the identified lines in wavelength_peak_positions, to select them from the outputs of find_lines(), e.g.
            wavelength_peak_positions[line_index], peak_heights[line_index], line_std_deviation[line_index].

        Notes
        -----
        A table with all the lines (identification, rest wavelength, velocity offset and status) is stored in self.line_identification, and
        the best alternative redshifts in self.redshift_candidates. Always check the identifications in the plot: a spectrum with a single
        line, or with lines that are not in the reference list, may have a wrong or ambiguous redshift.
        """

        observed = np.atleast_1d(self._values(wavelength_peak_positions)).astype(float)
        n_lines = len(observed)
        if n_lines == 0:
            raise ValueError("No lines were given.")
        significance = None if line_significance is None else np.atleast_1d(np.asarray(getattr(line_significance, "value", line_significance), dtype=float))
        if significance is not None and len(significance) != n_lines:
            raise ValueError("line_significance must have one value per line.")
        if line_types is None:
            if significance is not None:
                line_types = ["absorption" if s < 0 else "emission" for s in significance]
            elif getattr(self, "line_types", None) is not None and len(self.line_types) == n_lines:
                line_types = list(self.line_types)
            else:
                line_types = ["emission"]*n_lines
        kinds = [str(kind).lower() for kind in line_types]
        if any(kind not in ("emission", "absorption") for kind in kinds) or len(kinds) != n_lines:
            raise ValueError("line_types must be 'emission' or 'absorption', one per line.")
        if significance is None:
            significance = np.full(n_lines, 10.0)

        air_to_vacuum = line_identification.air_to_vacuum
        observed_vacuum = np.asarray(air_to_vacuum(observed), dtype=float)
        x_all = None if wavelengths is None else self._values(wavelengths)
        pixel = float(np.nanmedian(np.abs(np.diff(x_all)))) if x_all is not None else 0.0
        observed_range = (np.nanmin(x_all), np.nanmax(x_all)) if x_all is not None else (observed.min()/1.02, observed.max()*1.02)
        line_tolerance = np.maximum(float(tolerance), aux_identify_lines.SPEED_OF_LIGHT*2*pixel/observed)

        def table(entries):
            names, rest, weights = zip(*entries)
            rest = np.array(rest, dtype=float)
            return {"name": list(names), "rest": rest, "vacuum": aux_identify_lines.to_vacuum(rest, air_to_vacuum), "weight": np.array(weights, float)}
        references = {"emission": table(aux_identify_lines.EMISSION_REFERENCES), "absorption": table(aux_identify_lines.ABSORPTION_REFERENCES)}

        # Telluric absorption lines are not used for the redshift:
        telluric = [aux_identify_lines.telluric_band(w) if k == "absorption" else None for w, k in zip(observed, kinds)]
        usable = np.array([t is None for t in telluric])
        search_kinds = [k if u_ else "none" for k, u_ in zip(kinds, usable)]
        references["none"] = None

        if redshift is None:
            z_best, match, candidates = aux_identify_lines.find_redshift(observed_vacuum, significance, search_kinds, references, line_tolerance,
                                                                         observed_range, redshift_range)
        else:
            z_best = float(redshift)
            strength = np.log10(1 + np.abs(significance))
            match = aux_identify_lines._match(observed_vacuum, strength, search_kinds, z_best, references, line_tolerance)
            candidates = [(z_best, np.nan, int(np.count_nonzero(match >= 0)))]
        self.redshift_candidates = candidates

        # ------------------------------------------------------------ Names of all lines
        database_rest, database_names = aux_identify_lines.read_line_database(libpath / "astro_lines.dat")
        database_vacuum = aux_identify_lines.to_vacuum(database_rest, air_to_vacuum)
        rows = []
        for i in range(n_lines):
            row = {"index": i, "observed": observed[i], "significance": significance[i], "type": kinds[i], "name": None, "rest": np.nan,
                   "velocity": np.nan, "status": "unidentified"}
            if telluric[i] is not None:
                row["status"] = f"telluric ({telluric[i]})"
            elif z_best is not None and match[i] >= 0:
                reference = references[kinds[i]]
                j = match[i]
                row.update(name=reference["name"][j], rest=reference["rest"][j], status="identified",
                           velocity=aux_identify_lines.SPEED_OF_LIGHT*(observed_vacuum[i]/((1 + z_best)*reference["vacuum"][j]) - 1))
            elif z_best is not None and kinds[i] == "emission":
                velocity = aux_identify_lines.SPEED_OF_LIGHT*(observed_vacuum[i]/((1 + z_best)*database_vacuum) - 1)
                j = int(np.argmin(np.abs(velocity)))
                if abs(velocity[j]) <= line_tolerance[i]:
                    row.update(name=database_names[j], rest=database_rest[j], velocity=velocity[j], status="tentative")
            rows.append(row)

        # Intervening absorbers (pairs of absorption lines not explained at the redshift of the source):
        free = [row["index"] for row in rows if row["type"] == "absorption" and row["status"] == "unidentified"]
        z_max = (z_best if z_best is not None else redshift_range[1]) + 0.05
        for a, b, name, z_absorber in aux_identify_lines.intervening_doublets(observed, kinds, free, max(2*pixel, 1.0), air_to_vacuum, z_max):
            for k in (a, b):
                if rows[k]["status"] == "unidentified":
                    rows[k]["status"] = f"intervening {name} absorber (z = {z_absorber:.4f})"
                    rows[k]["name"] = name

        selected = [row for row in rows if row["status"] == "identified" or (use_tentative and row["status"] == "tentative")]
        line_index = np.array([row["index"] for row in selected], dtype=int)
        rest_frame_line_wavelengths = [float(row["rest"]) for row in selected]
        line_names = [row["name"] for row in selected]
        self.line_identification = Table(rows=[[r["index"], r["observed"], r["significance"], r["type"], r["name"] or "", r["rest"], r["velocity"],
                                                r["status"]] for r in rows],
                                         names=["index", "observed_wavelength", "significance", "type", "line", "rest_wavelength",
                                                "velocity_offset_km_s", "status"])

        # ------------------------------------------------------------ Summary
        if z_best is None:
            warnings.warn("No redshift could explain the detected lines with the reference lines. Identify the lines by hand (see the tutorial) "
                          "or give the 'redshift' if you know it.", stacklevel=3)
            z_value = np.nan
        else:
            z_value = float(z_best)
            n_matched = len(selected) if not use_tentative else sum(r["status"] == "identified" for r in rows)
            print(f"Estimated redshift: z = {z_value:.4f} ({n_matched} line(s) identified with the reference lines).")
            if redshift is None:
                if n_matched < 2:
                    warnings.warn("Only one line matches a reference line, so the redshift is ambiguous. Other possible redshifts: "
                                  + ", ".join(f"{z:.4f}" for z, _, _ in candidates[1:5]) + ". Check the plot, or give the 'redshift' if you know it.",
                                  stacklevel=3)
                elif len(candidates) > 1 and candidates[1][1] > 0.8*candidates[0][1]:
                    warnings.warn(f"The second best redshift (z = {candidates[1][0]:.4f}, {candidates[1][2]} lines) explains the lines almost as "
                                  "well as the best one. Check the plot.", stacklevel=3)
        print(f"{'#':>3} {'Observed (A)':>12} {'Signif.':>8}  {'Type':<10} {'Line':<14} {'Rest (A)':>9} {'dv (km/s)':>10}  Status")
        for r in rows:
            rest = f"{r['rest']:9.2f}" if np.isfinite(r["rest"]) else f"{'-':>9}"
            velocity = f"{r['velocity']:10.0f}" if np.isfinite(r["velocity"]) else f"{'-':>10}"
            print(f"{r['index']:>3} {r['observed']:12.1f} {abs(r['significance']):8.1f}  {r['type']:<10} {(r['name'] or '-'):<14} {rest} {velocity}  {r['status']}")

        # ------------------------------------------------------------ Plot
        if plot and x_all is not None and flux_density is not None:
            f_all = self._values(flux_density)
            good = np.isfinite(f_all)
            figure, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x_all, f_all, color="orange", linewidth=0.8, label=self.target_name if getattr(self, "target_name", None) else "Spectrum")
            top = np.nanpercentile(f_all[good], 99.8)
            line_heights = np.interp(observed, x_all, f_all)
            ax.set_ylim(min(0, np.nanpercentile(f_all[good], 0.2)), max(1.45*top, np.nanmax(line_heights) + 0.45*top))
            ax.set_xlim(x_all.min(), x_all.max())
            if z_best is not None:
                # Expected positions of the strong reference lines (emission) at the estimated redshift:
                reference = references["emission"]
                expected = reference["rest"]*(1 + z_value)
                shown = set(row["name"] for row in rows if row["status"] == "identified")
                for name, wavelength, weight in zip(reference["name"], expected, reference["weight"]):
                    if x_all.min() < wavelength < x_all.max() and weight >= 3 and name not in shown:
                        ax.axvline(wavelength, color="gray", linestyle=":", linewidth=0.8)
                        ax.text(wavelength, 0.97*ax.get_ylim()[1], name.replace("_", " "), rotation=90, fontsize=8, color="gray", ha="right", va="top")
            colors = {"identified": "C0", "tentative": "C4"}
            labelled = set()
            for r in rows:
                color = colors.get(r["status"], "C3" if r["status"].startswith(("intervening", "telluric")) else "black")
                height = np.interp(r["observed"], x_all, f_all)
                label = r["name"].replace("_", " ") if r["name"] else "?"
                if r["status"] == "tentative":
                    label += " (?)"
                elif r["status"].startswith("telluric"):
                    label = "telluric"
                elif r["status"].startswith("intervening"):
                    if r["status"] in labelled:
                        label = ""   # the doublet is labelled once, at its blue line
                    else:
                        labelled.add(r["status"])
                        label += " (z = " + r["status"].split("z = ")[1]
                ax.annotate(label, (r["observed"], height), xytext=(r["observed"], height + 0.12*top), rotation=90, fontsize=9, color=color,
                            ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6))
            handles = [plt.Line2D([], [], color="C0", label="Identified"), plt.Line2D([], [], color="C4", label="Tentative (line database)"),
                       plt.Line2D([], [], color="C3", label="Intervening absorber / telluric"), plt.Line2D([], [], color="black", label="Unidentified"),
                       plt.Line2D([], [], color="gray", linestyle=":", label="Other strong lines (expected)")]
            ax.legend(handles=[ax.get_lines()[0]] + handles, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
            ax.set_xlabel(r"Observed $\lambda$ [$\AA$]")
            ax.set_ylabel(r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]")
            ax.set_title(f"Line identification (z = {z_value:.4f})" if z_best is not None else "Line identification")
            ax.minorticks_on()
            ax.grid(which="both", linestyle=":", linewidth=0.5)
            figure.tight_layout()
            plt.show()

        return z_value, rest_frame_line_wavelengths, line_names, line_index

    # ==================================================================================================================================
    # Line fitting (the MCMC machinery is in aux_fit_lines.py)
    # ==================================================================================================================================

    def all_models(self, model_name):
        """
        Returns a picklable CombinedModel instance, e.g. all_models("GaussianLorentz") for a Gaussian plus a Lorentzian.
        """
        model_name_list = re.findall('[A-Z][a-z_]*', model_name)
        return CombinedModel(model_name_list)

    def lnlike(self, theta, x, y, yerr, model):
        """Log-likelihood of the model given the data (see aux_fit_lines.ln_likelihood)."""
        return ln_likelihood(theta, x, y, yerr, model)

    def lnprior(self, theta, priors):
        """Uniform priors (see aux_fit_lines.ln_prior)."""
        return ln_prior(theta, priors)

    def lnprob(self, theta, priors, x, y, yerr, model):
        """Log-posterior (see aux_fit_lines.ln_probability)."""
        return ln_probability(theta, priors, x, y, yerr, model)

    def line_MCMC(self, p0, priors, nwalkers, niter, initial, lnprob, data, model_name, custom_function=None, burn_in=100, show_progress=True):

        """
        Runs the MCMC for one line (single or blended) and a given model. Kept for compatibility: fit_lines() now uses
        aux_fit_lines.aux_run_mcmc() directly. 'p0' and 'lnprob' are ignored (the walkers start around the maximum a posteriori parameters).
        nwalkers, niter and burn_in can be integers or "auto".
        """

        x, y, yerr = data
        result = aux_run_mcmc(np.asarray(initial, dtype=float), np.asarray(priors, dtype=float), x, y, yerr, self.all_models(model_name),
                              walkers=nwalkers, iterations=niter, burn_in=burn_in, show_progress=show_progress, label=model_name)
        return result, result.flatchain[-1], result.flatlnprobability[-1], None

    def plotter(self, sampler, model_name, x, color="grey", normalization = 1):

        """In this function we plot the 'hairs' (i.e. alternative models drawn from the posterior distribution)."""

        samples = sampler.flatchain
        adopted_model = self.all_models(model_name)
        x_plot = np.linspace(x.min(), x.max(), 1000)
        for theta in samples[np.random.randint(len(samples), size=min(100, len(samples)))]:
            plt.plot(x_plot, adopted_model(theta, x_plot)*normalization, color=color, zorder=0, alpha=0.1)
        plt.grid(linestyle=":")

    def MCMC_spread(self, x, samples, model_name, nsamples=100, custom_function=None):

        """
        Median model and its standard deviation, from nsamples parameter sets drawn from the posterior distribution, evaluated on a grid of
        1000 points between x.min() and x.max().
        """

        x_plot = np.linspace(np.min(x), np.max(x), 1000)
        draw = np.random.randint(len(samples), size=nsamples)
        adopted_model = self.all_models(model_name)
        models = np.array([adopted_model(theta, x_plot) for theta in samples[draw]])
        return np.median(models, axis=0), np.std(models, axis=0)

    def data_window_selection(self, wavelengths, spec_flux, spec_flux_err, line_region_min, line_region_max):

        """
        Function to slice the data into a wavelength window.

        Parameters
        ----------
        wavelengths: numpy.array
            This variable must be in absolute values, i.e. without units.
        spec_flux: numpy.array
            The spectral density array.
        spec_flux_err: numpy.array
            The uncertainty of the spectral density.
        line_region_min: float
            Inferior window limit.
        line_region_max: float
            Superior window limit.

        Returns
        -------
        x, y, yerr: numpy.arrays
            New arrays containing data within the selected window.
        """

        selection = (wavelengths > line_region_min) & (wavelengths < line_region_max)
        return wavelengths[selection], spec_flux[selection], spec_flux_err[selection]

    def redshift_calculator(self, q_16, q_50, q_84, air_wavelength_line):
        """Redshift and its errors from the quantiles of the observed wavelength (same medium for both wavelengths)."""
        z = (q_50 - air_wavelength_line)/air_wavelength_line
        return z, (q_50 - q_16)/air_wavelength_line, (q_84 - q_50)/air_wavelength_line

    def parameter_estimation(self, samples, air_wavelength_line=None, quantiles=[0.16, 0.5, 0.84], normalization = 1, parlabels=None, line_names="",
                             output_dir=".", savefile=True):

        """
        Estimate the parameter values (median) and their asymmetric errors (16% and 84% quantiles) from the MCMC posterior distribution.
        fit_lines() uses aux_fit_lines.aux_summarize_posterior() directly; this function is kept for compatibility.

        Parameters
        ----------
        samples: numpy.ndarray
            The MCMC posterior samples (n_samples x n_parameters).
        air_wavelength_line: float or list
            The rest wavelength(s) of the line(s), used to compute the redshift (in the same medium as the observed wavelengths).
        quantiles: list
            The quantiles adopted for the value and the errors. Default is [0.16, 0.5, 0.84].
        normalization: float
            The amplitudes are multiplied by this factor.
        parlabels: list
            The names of the parameters of the model, or a list of lists for blended lines, e.g. [['Location','Amplitude','fwhm_Gauss'], [...]].
        line_names: string or list of strings
            The line names.
        output_dir: string
            The output directory.
        savefile: boolean
            If True, the results are saved in "<target>_<line>_line_fit_results.csv" in the output directory.

        Returns
        -------
        par_values, par_values_errors, par_names: lists
        """

        if isinstance(parlabels[0], str):
            parlabels = [parlabels]
        if air_wavelength_line is None:
            rest = [None]*len(parlabels)
        else:
            rest = list(np.atleast_1d(air_wavelength_line))
        line_names = [line_names] if isinstance(line_names, str) else list(line_names)
        values, errors, names = aux_summarize_posterior(np.asarray(samples), parlabels, rest, normalization, quantiles=quantiles)
        par_values, par_values_errors, par_names = [], [], []
        for number, (v, e, n) in enumerate(zip(values, errors, names)):
            par_values += v
            par_values_errors += e
            par_names += n
            if savefile:
                self._write_line_results(line_names[number], v, e, n, output_dir)
        return par_values, par_values_errors, par_names

    def _write_line_results(self, line_name, values, errors, names, output_dir=None):
        """Write the fit results of one line to "<target>_<line>_line_fit_results.csv" and return the path."""
        path = Path(self.output_dir if output_dir is None else output_dir) / f"{self.target_name}_{line_name}_line_fit_results.csv"
        with open(path, "w") as results_file:
            results_file.write("# Parameter name, value, error_down, error_up\n")
            for name, value, (down, up) in zip(names, values, errors):
                results_file.write(f"{'z' if name == 'redshift' else name}, {value}, {down}, {up}\n")
        return path

    @_plot_style
    def quick_plot(self, x, y, model_name, parlabels, sampler=None, best_fit_model=None, theta_max=None, normalization = 1, hair_color="grey", title="",
                   xlabel=r"Observed $\lambda$ [$\AA$]", ylabel=r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ ]", overplot_median_model=True,
                   savefig=True, outputdir="./", yerr=None):

        """
        In this function we plot the data window together with the maximum likelihood and/or the median model.

        Parameters
        ----------
        x, y: numpy.arrays
            Arrays containing the data within the selected window.
        model_name: string
            The current model adopted in the fit.
        parlabels: list
            A list with the parameter names for the adopted model.
        sampler:
            The MCMC result (or emcee sampler).
        best_fit_model: numpy.array
            Array with the best-fit model, evaluated on a grid between x.min() and x.max().
        theta_max: numpy.ndarray
            Array with the best-fit parameters.
        normalization: float
            The plot normalization.
        hair_color: string
            The color adopted for the MCMC hairs.
        title: string
            The title of the plot.
        xlabel,ylabel: strings
            The axes labels.
        overplot_median_model: boolean
            If True, the median model and model standard deviation will be overplotted.
        savefig: boolean
            If True, the figure will be saved in the output directory.
        outputdir: string
            The output directory.
        yerr: numpy.array
            Optional. Uncertainties of the data, shown as a shaded band.
        """

        figure = plt.figure(figsize=(10,8))
        ax = figure.add_subplot(1,1,1)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='major', length=5, direction='in')
        ax.tick_params(which='minor', length=2.5, direction='in', bottom=True, top=True, left=True, right=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)

        if yerr is not None:
            plt.fill_between(x, (y - yerr)*normalization, (y + yerr)*normalization, color="orange", alpha=0.25, step="mid", linewidth=0)
        plt.plot(x, y*normalization, color="orange", label="Data after cont. subtraction")
        if sampler is not None:
            self.plotter(sampler, model_name, x, color=hair_color, normalization=normalization)
        if best_fit_model is not None:
            x_plot = np.linspace(x.min(), x.max(), len(best_fit_model))
            plt.plot(x_plot, best_fit_model*normalization, color="black", label="Highest likelihood model")
        if overplot_median_model and sampler is not None:
            median_model, spread = self.MCMC_spread(x, sampler.flatchain, model_name, nsamples=200)
            x_plot = np.linspace(x.min(), x.max(), len(median_model))
            plt.plot(x_plot, median_model*normalization, color="C0", ls="--", label="Median model")
            plt.fill_between(x_plot, (median_model + spread)*normalization, (median_model - spread)*normalization, color="C0", alpha=0.3,
                             label=r"$1\sigma$")

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title)
        plt.grid(which="both", linestyle=":")
        plt.ticklabel_format(axis="y", scilimits=(-5, 8))
        low, high = np.nanmin(y)*normalization, np.nanmax(y)*normalization
        margin = 0.1*(high - low)
        plt.ylim(low - margin, high + margin)
        if theta_max is not None:
            for i in np.nonzero(np.asarray(parlabels) == "Location")[0]:
                plt.axvline(theta_max[i], color="k", linewidth=0.5)
        plt.legend()
        if savefig:
            safe_title = re.sub(r"[^\w\-+.]", "_", title)
            plt.savefig(Path(outputdir) / f"{safe_title}_line.png", bbox_inches="tight")

    def insert_rows(self, array, indices, row):
        """Insert one or more copies of 'row' into 'array' at given indices (axis=0)."""
        array = np.asarray(array)
        row = np.asarray(row)
        for idx in np.clip(np.sort(indices)[::-1], 0, len(array)):
            array = np.insert(array, idx, row, axis=0)
        return array

    def merge_fit_results(self, target_name, list_of_files=None, output_dir="./"):

        """
        This function merges the individual data files of each line ("<target>_<line>_line_fit_results.csv") into a single file,
        "<target>_lines.csv". Parameters that a line does not have (e.g. the skewness of a Gaussian) are written as nan.

        Parameters
        ----------
        target_name: string
            The target name (prefix of the files).
        list_of_files: list or string
            Optional. The files to merge (a list of paths, or a text file with one path per line). Default: all the files
            "<target>_*_line_fit_results.csv" in output_dir.
        output_dir: string
            The output directory.
        """

        output_dir = Path(output_dir)
        suffix = "_line_fit_results.csv"
        if list_of_files is None:
            list_of_files = sorted(output_dir.glob(f"{target_name}_*{suffix}"))
        elif isinstance(list_of_files, (str, Path)):
            list_of_files = list(np.atleast_1d(np.genfromtxt(list_of_files, dtype=str)))
        list_of_files = [Path(f) for f in list_of_files if not str(f).endswith("custom_model.csv")]
        if not list_of_files:
            warnings.warn(f"No '*{suffix}' files found to merge in {output_dir}.")
            return

        tables, line_names = [], []
        for line_file in list_of_files:
            rows = np.atleast_2d(np.genfromtxt(line_file, delimiter=",", dtype=str, comments="#", autostrip=True))
            tables.append({row[0]: row[1:4] for row in rows})
            name = line_file.name
            prefix = f"{target_name}_"
            line_names.append(name[len(prefix):-len(suffix)] if name.startswith(prefix) and name.endswith(suffix) else name[:-len(suffix)])

        parameter_names = []
        for name in ['z', 'Location', 'Amplitude', 'fwhm_Gauss', 'fwhm_Lorentz', 'skewness']:
            if name not in parameter_names:
                parameter_names.append(name)
        for table in tables:   # any other parameter, in order of appearance
            for name in table:
                if name not in parameter_names:
                    parameter_names.append(name)

        units = {"Location": " [Ang]", "Amplitude": " (flux_dens - continuum) [erg cm-2 s-1 Ang-1]", "fwhm_Gauss": " [Ang]", "fwhm_Lorentz": " [Ang]"}
        with open(output_dir / f"{target_name}_lines.csv", "w") as merged:
            merged.write("# Line name")
            for name in parameter_names:
                prefix = "obs_" if name in ("Location", "Amplitude", "fwhm_Gauss", "fwhm_Lorentz", "skewness") else ""
                merged.write(f", {prefix}{name}{units.get(name, '')}, {prefix}{name}_error_down, {prefix}{name}_error_up")
            if self.wavelength_systematic_error is not None:
                merged.write(", Systematic wavelength error [Ang]")
            if self.flux_err is not None:
                merged.write(", Flux uncertainty at the line peak [erg cm-2 s-1 Ang-1]")
            for line_name, table in zip(line_names, tables):
                merged.write("\n" + line_name)
                for name in parameter_names:
                    value, down, up = table.get(name, ["nan", "nan", "nan"])
                    merged.write(f", {value}, {abs(float(down))}, {abs(float(up))}")
                if self.wavelength_systematic_error is not None:
                    merged.write(f", {self.wavelength_systematic_error.value}")
                if self.flux_err is not None and "Location" in table and self.wavelengths is not None:
                    index = extraction.find_nearest(self.wavelengths.value, float(table["Location"][0]))
                    merged.write(f", {self.flux_err.value[index]}")
                elif self.flux_err is not None:
                    merged.write(", nan")
            merged.write("\n")

    @_plot_style
    def parameter_time_series(self, initial, sampler, labels):

        """
        This function plots the time series of the parameters of all the walkers (including the burn-in, marked by a dashed line).
        """

        chain = sampler.get_chain()
        n_parameters = chain.shape[2]
        fig, axes = plt.subplots(n_parameters, figsize=(10, 2*n_parameters), sharex=True, squeeze=False)
        burn_in = getattr(sampler, "info", {}).get("burn_in")
        for i in range(n_parameters):
            ax = axes[i, 0]
            ax.plot(chain[:, :, i], "k", alpha=0.2, linewidth=0.5)
            ax.set_xlim(0, len(chain))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            if burn_in:
                ax.axvline(burn_in, color="C3", linestyle="--", linewidth=1)
        axes[-1, 0].set_xlabel("Step")

    def automatic_priors(self, which_model, observed_wavelength, peak_height, line_region_min, line_region_max):

        """
        Automatic initial values and priors of a line (kept for compatibility; fit_lines() uses aux_fit_lines.aux_automatic_priors()).
        The FWHM is initialized at 10 Angstroms and allowed between 0.1 Angstrom and the width of the window.

        Returns
        -------
        initial, priors, labels, adopted_model
        """

        if line_region_min is None or line_region_max is None:
            line_region_min, line_region_max = observed_wavelength - 100, observed_wavelength + 100
        initial, priors, labels = aux_automatic_priors(which_model, observed_wavelength, peak_height, 10.0, [line_region_min, line_region_max],
                                                       [0.1, max(1.0, line_region_max - line_region_min)])
        return initial, priors, labels, self.all_models(which_model)

    def estimate_norm_height(self, flux_density, continuum_baseline, wavelength_peak_position, peak_height):

        """
        Normalized height of a peak, i.e. divided by the normalization used in the MCMC (a power of 10 of the order of the flux density).
        """

        return peak_height/self._normalization(flux_density, continuum_baseline)

    @staticmethod
    def _normalization(flux_density, continuum_baseline=None):
        """A power of ten of the order of the flux density (keeps the MCMC parameters of order unity)."""
        values = np.abs(np.asarray(getattr(flux_density, "value", flux_density), dtype=float))
        level = np.nanmedian(values[values > 0]) if np.any(values > 0) else 1.0
        return 10.0**np.round(np.log10(level))

    def calculate_continuum_subtracted_heights(self, wavelengths, peak_positions, peak_heights, continuum_baseline):
        """Calculate peak heights relative to local continuum."""
        continuum_indices = [extraction.find_nearest(self._values(wavelengths), pos) for pos in peak_positions]
        return peak_heights - np.asarray(continuum_baseline)[continuum_indices]

    def fit_lines(self, wavelengths, flux_density, continuum_baseline, wavelength_peak_positions, rest_frame_line_wavelengths, peak_heights, line_std_deviation,
                  blended_line_min_separation = 50, which_models = "Lorentz", line_names = None, overplot_archival_lines = ["H"], priors = None,
                  MCMC_walkers = "auto", MCMC_iterations = "auto", MCMC_burn_in = "auto", MCMC_show_progress = True, plot_spec = True, plot_MCMC = False,
                  overplot_median_model = False, save_results = True, rest_frame_medium = "auto", rescale_errors = True, MCMC_max_steps = 20000):

        """
        Fit the spectral lines with a Markov Chain Monte Carlo (MCMC), to estimate the line parameters and their uncertainties.

        For each line (or group of blended lines), easyspec:
        1. selects a window around the line (at least +-100 A, or 3 times the estimated FWHM of the line, limited by the neighbouring lines);
        2. sets the initial values and the (uniform) priors, from the detected peak position and height and from the estimated FWHM;
        3. finds the best fit (maximum a posteriori), rescales the uncertainties of the data if the scatter around the best fit is larger than
           expected (see 'rescale_errors'), and runs the MCMC starting around the best fit, with the number of walkers, burn-in and chain
           length chosen automatically from the autocorrelation time of the chain (or given by the user);
        4. computes the median and the 16% and 84% quantiles of each parameter, and the redshift of each line.

        The uncertainty of each pixel is the statistical uncertainty of the spectrum (flux_err, read by load_calibrated_data()), or the noise of
        the local continuum (line_std_deviation) if it is not available. Pixels flagged in the data quality mask are not used.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        continuum_baseline: numpy.ndarray (float)
            The continuum flux density (output of analysis.find_lines()).
        wavelength_peak_positions: numpy.ndarray (astropy.units Angstrom)
            The position of each line (output of analysis.find_lines(), or given by hand).
        rest_frame_line_wavelengths: list
            The rest-frame wavelength of each line, in Angstroms (see 'rest_frame_medium').
        peak_heights: numpy.ndarray (astropy.units erg/cm2/s/A)
            The flux density at each peak (output of analysis.find_lines()).
        line_std_deviation: numpy.ndarray (float)
            The noise of the local continuum at each line (output of analysis.find_lines()).
        blended_line_min_separation: float
            Lines closer than this separation (in Angstroms) are fitted together (blended model).
        which_models: string or list of strings
            The model of each line, e.g. which_models = ["Lorentz", "Gaussian"], or a single string for all lines. Options are "Gaussian",
            "Lorentz", "Voigt", "skewed_gaussian" and "skewed_lorentzian". All widths are FWHMs.
        line_names: list
            Optional. The line names, e.g. ["Hbeta", "Halpha"]. Default: line_0, line_1...
        overplot_archival_lines: list
            Elements whose lines (from the easyspec list, redshifted with the average redshift of the fitted lines) are overplotted on the
            spectrum, e.g. ["H", "He"]. Use None to overplot no lines.
        priors: list
            Optional. User-defined priors for some lines: one entry per line, None for the automatic priors, or a list with one [min, max] range
            per parameter of the model of that line, in the order: Location (Angstrom), Amplitude (in units of the normalization of the
            spectrum, i.e. a power of 10 printed by fit_lines()), then the FWHM(s) in Angstroms ("fwhm_Lorentz" and then "fwhm_Gauss" for the
            Voigt model) and the skewness (skewed models). E.g. for 4 Lorentzian lines: priors = [None, None, [[7496,7696],[0.1,150],[2,150]],
            None]. The location range also defines the fit window of that line.
        MCMC_walkers: "auto" or int
            Number of walkers. "auto" (default): max(32, 8 x number of parameters).
        MCMC_iterations: "auto" or int
            Number of steps after the burn-in. "auto" (default): the chain runs until it is longer than 50 autocorrelation times and the
            autocorrelation time is stable (changes < 5%), up to MCMC_max_steps.
        MCMC_burn_in: "auto" or int
            Number of initial steps discarded. "auto" (default): 3 times the autocorrelation time.
        MCMC_show_progress: boolean
            If True, the progress bars of the MCMC are shown.
        plot_spec: boolean
            If True, the spectrum is plotted with the lines requested in overplot_archival_lines.
        plot_MCMC: boolean
            If True, the diagnostic plots of the MCMC are shown: corner plot, time series of the parameters, and the fit.
        overplot_median_model: boolean
            If True, the median model and its standard deviation are overplotted in the fit plots.
        save_results: boolean
            If True, the plots and the file "<target>_lines.csv" with the results are saved in the output directory.
        rest_frame_medium: string
            Medium of rest_frame_line_wavelengths: "air", "vacuum", or "auto" (default; vacuum below 2000 A and air above, the usual convention
            and the one of the easyspec line list). The observed wavelengths (air) and the rest wavelengths are converted to vacuum to compute the
            redshifts.
        rescale_errors: boolean
            If True (default), the uncertainties of the data are multiplied by sqrt(reduced chi2) of the best fit when it is larger than 1, so that
            the errors of the parameters account for the actual scatter of the data around the model.
        MCMC_max_steps: int
            Maximum number of steps with MCMC_iterations = "auto". Default is 20000.

        Returns
        -------
        par_values_list: list
            For each line, the best-fit values: [redshift, Location, Amplitude, widths...].
        par_values_errors_list: list
            The asymmetric errors [down, up] of each value in par_values_list.
        par_names_list: list
            The names of the values in par_values_list.
        samples_list: list
            The MCMC posterior samples of each fit (one per line or group of blended lines; amplitudes normalized).
        line_windows: list
            The wavelength window used for each line.

        Notes
        -----
        The MCMC diagnostics of each fit (walkers, steps, burn-in, autocorrelation time, acceptance fraction, reduced chi2, error scale) are
        stored in self.fit_info, and the posterior samples of each line (used by line_physics()) in self.line_posteriors.
        """

        # ---------------------------------------------------------------- Inputs
        aux_validate_line_names(line_names)
        x_all = self._values(wavelengths)
        f_all = self._values(flux_density)
        flux_unit = getattr(flux_density, "unit", FLAM_UNIT)
        continuum_baseline = np.asarray(continuum_baseline, dtype=float)
        peaks = aux_ensure_numpy_array(wavelength_peak_positions, 'wavelength_peak_positions')
        heights = aux_ensure_numpy_array(peak_heights, 'peak_heights')
        noise_at_lines = aux_ensure_numpy_array(line_std_deviation, 'line_std_deviation')
        rest = aux_ensure_numpy_array(rest_frame_line_wavelengths, 'rest_frame_line_wavelengths')
        n_lines = len(rest)
        if not (len(peaks) == len(heights) == len(noise_at_lines) == n_lines):
            raise ValueError(f"The inputs must have one value per line: {len(peaks)} peak positions, {len(heights)} peak heights, "
                             f"{len(noise_at_lines)} noise values and {n_lines} rest-frame wavelengths.")
        order = np.argsort(peaks)
        if not np.array_equal(order, np.arange(n_lines)):
            raise ValueError("The lines must be given in increasing order of wavelength.")
        models = aux_validate_models(which_models, n_lines)
        line_names = aux_generate_line_names(line_names, n_lines)
        self.line_names = line_names
        if priors is None:
            priors = [None]*n_lines
        elif n_lines == 1 and priors is not None and len(priors) > 0 and priors[0] is not None and np.ndim(priors[0]) == 1:
            priors = [priors]   # a single line given as priors=[[7500,7700],[0.1,10],[2,50]]
        if len(priors) != n_lines:
            raise ValueError(f"'priors' must have one entry (None or a list of ranges) per line ({n_lines}).")
        if rest_frame_medium not in ("auto", "air", "vacuum"):
            raise ValueError("'rest_frame_medium' must be 'auto', 'air' or 'vacuum'.")

        def rest_to_vacuum(wavelength):
            if rest_frame_medium == "vacuum" or (rest_frame_medium == "auto" and wavelength < 2000):
                return wavelength
            return float(line_identification.air_to_vacuum(wavelength))

        heights = self.calculate_continuum_subtracted_heights(x_all, peaks, heights, continuum_baseline)
        good = self._good_pixels(x_all, f_all)
        flux_err = self._matching(self.flux_err, len(f_all))
        residual_all = f_all - continuum_baseline
        pixel = np.nanmedian(np.abs(np.diff(x_all)))
        normalization = self._normalization(flux_density)
        print(f"Normalization of the amplitudes in the MCMC: {normalization:.0e} {flux_unit}")

        # Estimated FWHM of each line (initial value, priors and fit window):
        fwhm_estimates = []
        for peak in peaks:
            fwhm = aux_estimate_fwhm(x_all, np.where(good, residual_all, 0.0), extraction.find_nearest(x_all, peak))
            fwhm_estimates.append(fwhm if np.isfinite(fwhm) and fwhm > 0 else 10.0)
        fwhm_estimates = np.array(fwhm_estimates)

        groups = aux_group_blended_lines(peaks, blended_line_min_separation)
        par_values_list, par_values_errors_list, par_names_list, samples_list, line_windows = [], [], [], [], []
        self.fit_info = []
        self.line_posteriors = []   # Posterior samples of each line, used by line_physics()
        if save_results:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        written_files = []

        for group in groups:
            first, last = group[0], group[-1]
            # ---------------------------------------------------------------- Window of the group
            lower = min(peaks[i] - max(100.0, 3*fwhm_estimates[i]) for i in group)
            upper = max(peaks[i] + max(100.0, 3*fwhm_estimates[i]) for i in group)
            if first > 0:
                lower = max(lower, 0.5*(peaks[first - 1] + peaks[first]))
            if last < n_lines - 1:
                upper = min(upper, 0.5*(peaks[last] + peaks[last + 1]))
            for i in group:   # user-defined location priors define the window
                if priors[i] is not None:
                    location_low, location_high = float(np.min(priors[i][0])), float(np.max(priors[i][0]))
                    if len(group) == 1:
                        lower, upper = location_low, location_high
                    else:
                        lower, upper = min(lower, location_low), max(upper, location_high)

            # ---------------------------------------------------------------- Initial values and priors of each line of the group
            initial, bounds, labels = [], [], []
            for position, i in enumerate(group):
                height = heights[i]/normalization
                if priors[i] is not None:
                    line_initial, line_bounds = aux_parse_custom_priors(priors[i], models[i], peaks[i], height)
                    line_labels = list(MODEL_PARAMETERS[models[i]])
                else:
                    # In a blend, each line is confined between the midpoints to its neighbours, which avoids that two components swap.
                    location_low = lower if position == 0 else 0.5*(peaks[group[position - 1]] + peaks[i])
                    location_high = upper if position == len(group) - 1 else 0.5*(peaks[i] + peaks[group[position + 1]])
                    line_initial, line_bounds, line_labels = aux_automatic_priors(models[i], peaks[i], height, fwhm_estimates[i],
                                                                                  [location_low, location_high],
                                                                                  [max(0.5*pixel, 0.05), max(4*pixel, upper - lower)])
                initial.append(line_initial)
                bounds.append(line_bounds)
                labels.append(line_labels)
            initial, bounds = np.concatenate(initial), np.concatenate(bounds)
            model_name = "".join(models[i] for i in group)
            adopted_model = self.all_models(model_name)

            # ---------------------------------------------------------------- Data of the window
            selection = (x_all > lower) & (x_all < upper) & good
            x, y = x_all[selection], residual_all[selection]/normalization
            if flux_err is not None:
                yerr = flux_err[selection]/normalization
                yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nanmedian(yerr))
            else:
                yerr = np.full(len(x), np.median(noise_at_lines[group])/normalization)
            if len(x) <= len(initial):
                raise RuntimeError(f"The window {lower:.1f}-{upper:.1f} A has only {len(x)} valid pixels, which is not enough to fit "
                                   f"{' + '.join(line_names[i] for i in group)}.")

            # ---------------------------------------------------------------- MCMC
            group_label = "+".join(line_names[i] for i in group)
            result = aux_run_mcmc(initial, bounds, x, y, yerr, adopted_model, walkers=MCMC_walkers, iterations=MCMC_iterations,
                                  burn_in=MCMC_burn_in, max_steps=MCMC_max_steps, rescale_errors=rescale_errors,
                                  show_progress=MCMC_show_progress, label=group_label)
            samples = result.flatchain
            samples_list.append(samples)
            theta_max = samples[np.argmax(result.flatlnprobability)]
            self.fit_info.append(dict(result.info, lines=[line_names[i] for i in group], window=[lower, upper], model=model_name))

            values, errors, names = aux_summarize_posterior(samples, labels, [rest[i] for i in group], normalization,
                                                            observed_to_vacuum=line_identification.air_to_vacuum,
                                                            rest_to_vacuum=rest_to_vacuum)
            # Posterior samples of each line (amplitudes in physical units), and of its redshift:
            physical = samples.copy()
            column = 0
            columns = []
            for line_labels in labels:
                columns.append(np.arange(column, column + len(line_labels)))
                physical[:, column + 1] *= normalization   # Amplitude
                column += len(line_labels)
            for position, i in enumerate(group):
                location_samples = physical[:, columns[position][0]]
                z_samples = line_identification.air_to_vacuum(location_samples)/rest_to_vacuum(rest[i]) - 1
                self.line_posteriors.append({"line": line_names[i], "model": models[i], "labels": labels[position],
                                             "samples": physical[:, columns[position]], "z_samples": z_samples, "rest_wavelength": rest[i],
                                             "group": [line_names[j] for j in group], "group_models": [models[j] for j in group],
                                             "group_samples": [physical[:, column] for column in columns], "window": [lower, upper],
                                             "error_scale": result.info["error_scale"]})
            for i, v, e, n in zip(group, values, errors, names):
                par_values_list.append(v)
                par_values_errors_list.append(e)
                par_names_list.append(n)
                line_windows.append([lower, upper])
                if save_results:
                    written_files.append(self._write_line_results(line_names[i], v, e, n))

            if plot_MCMC:
                flat_labels = [f"{label} ({line_names[i]})" if len(group) > 1 else label for i, line_labels in zip(group, labels) for label in line_labels]
                corner.corner(samples, show_titles=True, labels=flat_labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
                plt.suptitle(group_label + "\n(normalized amplitude)", x=0.7)
                self.parameter_time_series(initial, result, flat_labels)
                x_best_fit = np.linspace(x.min(), x.max(), 1000)
                self.quick_plot(x, y, model_name=model_name, parlabels=[label for line_labels in labels for label in line_labels], sampler=result,
                                best_fit_model=adopted_model(theta_max, x_best_fit), theta_max=theta_max, normalization=normalization,
                                hair_color="grey", title=f"{self.target_name} - {group_label}",
                                ylabel=r"F$_{\lambda} - F_{continuum}$ " + f"[{flux_unit}]", overplot_median_model=overplot_median_model,
                                savefig=save_results, outputdir=self.output_dir, yerr=result.info["error_scale"]*yerr)

        # ---------------------------------------------------------------- Redshift summary and plot
        redshifts = np.array([values[0] for values in par_values_list])
        redshift_errors = np.array([np.mean(errors[0]) for errors in par_values_errors_list])
        weights = np.where(redshift_errors > 0, 1/redshift_errors**2, 0)
        average_redshift = float(np.sum(weights*redshifts)/np.sum(weights)) if np.sum(weights) > 0 else float(np.mean(redshifts))
        std_redshift = float(np.std(redshifts))
        print("Redshifts: " + ", ".join(f"{n} {z:.5f}" for n, z in zip(line_names, redshifts))
              + f"; weighted mean = {average_redshift:.5f}, scatter = {std_redshift:.5f}")

        if plot_spec:
            with plt.rc_context({"font.size": 12}):
                top = np.nanpercentile(f_all[good], 99.8)
                plt.figure(figsize=(12,5))
                if overplot_archival_lines is not None:
                    print("Archival lines are taken from NIST Atomic Spectra database: https://www.nist.gov/pml/atomic-spectra-database")
                    print("We adopt vacuum wavelengths for lines with wavelengths < 2000 Angstroms and air wavelengths for lines with wavelengths > 2000 Angstroms.")
                    print("If you use these lines in your research, please cite the NIST Atomic Spectra database appropriately.")
                    archival_lines = np.loadtxt(libpath / "astro_lines.dat", dtype=str, delimiter=",")
                    archival_wavelengths = archival_lines[:,0].astype(float)*(1 + average_redshift)
                    archival_line_names = archival_lines[:,1]
                    for wavelength, name in zip(archival_wavelengths, archival_line_names):
                        if not x_all.min() < wavelength < x_all.max():
                            continue
                        element = name.split("_")[0].lstrip("[")
                        if element not in overplot_archival_lines:
                            continue
                        split_name = name.split("_")
                        label = split_name[0] + " " + split_name[1] + fr"$\{split_name[2]}$" if len(split_name) == 3 else "".join(split_name[:2])
                        color, step = {"H": ("C0", 1.1), "He": ("C0", 1.1), "O": ("C2", 1.22), "N": ("C3", 1.4), "S": ("C4", 1.6)}.get(element, ("black", 1.1))
                        plt.text(wavelength, step*top, label, rotation=90, fontsize=10, color=color, ha="center", va="bottom")
                        plt.vlines(wavelength, f_all[extraction.find_nearest(x_all, wavelength)], 0.98*step*top, color=color, linewidth=0.8, alpha=0.5)
                plt.plot(x_all, continuum_baseline, label="Continuum")
                plt.plot(x_all, f_all, color='orange', linewidth=0.8)
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                plt.xlim(x_all.min(), x_all.max())
                plt.ylim(min(0, np.nanpercentile(f_all[good], 0.5)), 2*top)
                plt.ylabel(r"F$_{\lambda}$ " + f"[{flux_unit}]", fontsize=12)
                plt.xlabel(r"Observed $\lambda$ [$\AA$]", fontsize=12)
                plt.title(f"{self.target_name} - " + r"$z_{av}$" + f" = {average_redshift:.5f}, " + r"$\sigma_z$" + f" = {std_redshift:.5f}")
                plt.legend()
                plt.tight_layout()
                if save_results:
                    plt.savefig(Path(self.output_dir) / f"{self.target_name}_spec.pdf", bbox_inches='tight')

        if save_results:
            self.merge_fit_results(self.target_name, list_of_files=written_files, output_dir=self.output_dir)
            for file in written_files:   # only the files written by this call are removed
                Path(file).unlink(missing_ok=True)

        if plot_spec or plot_MCMC:
            plt.show()

        return par_values_list, par_values_errors_list, par_names_list, samples_list, line_windows


    
    # ==================================================================================================================================
    # Physical properties of the lines (the computations are in aux_line_physics.py)
    # ==================================================================================================================================

    @_plot_style
    def line_dispersion_and_equiv_width(self, wavelengths_window, flux_density_window, continuum_baseline_window, line_fit_parameters,
                                        line_std_deviation, line_name = None, plot = True, flux_err_window = None):

        """
        Model-independent line dispersion (rms width), equivalent width and FWHM of a line, measured on the interpolated (continuum-subtracted)
        line profile, with Monte Carlo uncertainties. line_physics() uses the same method; this function is kept for compatibility.

        Parameters
        ----------
        wavelengths_window: numpy.ndarray
            The wavelength window for the given line.
        flux_density_window: numpy.ndarray
            The flux density window for the given line.
        continuum_baseline_window: numpy.ndarray
            The continuum flux density in the window.
        line_fit_parameters: list
            The best-fit values of the line, [redshift, Location, Amplitude, ...]; the location is used to find the line in the window.
        line_std_deviation: float
            The noise of the local continuum (output of analysis.find_lines()), used as the uncertainty of each pixel if flux_err_window is None,
            and to define the integration limits (where the profile reaches the noise level).
        line_name: string
            The line name.
        plot: boolean
            If True, the line and its centroid, dispersion and FWHM are shown.
        flux_err_window: numpy.ndarray
            Optional. The 1-sigma uncertainty of each pixel of the window.

        Returns
        -------
        line_dispersion, profile_equiv_width, profile_FWHM: floats
            Observed-frame line dispersion (second moment of the profile; see Eqs. 4 and 5 in Peterson et al. 2004), equivalent width
            (integral of (Fc - F)/Fc; positive for absorption lines) and FWHM.
        line_disp_error, equiv_width_error, fwhm_error: floats
            Their 1-sigma uncertainties (mean of the lower and upper errors).
        """

        x = self._values(wavelengths_window)
        flux = self._values(flux_density_window)
        continuum = self._values(continuum_baseline_window)
        y = flux - continuum
        error = np.full(len(x), float(line_std_deviation)) if flux_err_window is None else self._values(flux_err_window)
        peak = extraction.find_nearest(x, line_fit_parameters[1])
        left, right = aux_line_physics.integration_limits(x, y, error, peak, x.min(), x.max())
        self.integration_limits_dispersion = [x[left], x[right]]
        result = aux_line_physics.profile_physics(x[left:right + 1], y[left:right + 1], error[left:right + 1], continuum[left:right + 1], 0.0)
        if plot:
            self._plot_line_profile(x, y, None, left, right, result, line_name)
        mean_error = lambda v: 0.5*(v[1] + v[2])
        return (result["dispersion_rest"][0], result["equivalent_width_rest"][0], result["fwhm_rest"][0], mean_error(result["dispersion_rest"]),
                mean_error(result["equivalent_width_rest"]), mean_error(result["fwhm_rest"]))

    def line_dispersion_skewed_models(self, par_values, theta_error, model = "Lorentz"):

        """
        Line dispersion and FWHM of a skewed line model, with uncertainties from parameter samples drawn from the (asymmetric) errors. Kept for
        compatibility: line_physics() uses the MCMC posterior samples directly.

        Parameters
        ----------
        par_values: list
            The best-fit values [redshift, Location, Amplitude, FWHM, skewness].
        theta_error: numpy.ndarray
            The asymmetric errors [[down, up], ...] of [Location, Amplitude, FWHM, skewness].
        model: string
            "Gauss" or "Lorentz".

        Returns
        -------
        line_dispersion, model_FWHM, line_disp_error, fwhm_error: floats (observed frame). The dispersion of the skewed Lorentzian is np.nan
        (its second moment diverges).
        """

        model_name = "Skewedlorentzian" if model.lower().startswith("lorentz") else "Skewedgaussian"
        theta = np.asarray(par_values[1:], dtype=float)
        _, fwhm, dispersion = aux_line_physics.model_quantities(model_name, theta)
        samples = aux_line_physics.split_normal_samples(theta, theta_error, 300, np.random.default_rng())
        values = np.array([aux_line_physics.model_quantities(model_name, s) for s in samples])
        return dispersion, fwhm, float(np.nanstd(values[:, 2])) if np.any(np.isfinite(values[:, 2])) else np.nan, float(np.nanstd(values[:, 1]))

    def error_propagation_voigt(self, FWHM_Lorentz, FWHM_Gauss, error_lorentz, error_gauss):

        """
        Propagated uncertainty of the Voigt FWHM (Olivero & Longbothum 1977 approximation), assuming independent errors of the FWHMs of the
        Lorentzian and Gaussian components. (line_physics() uses the MCMC samples, which include their correlation.)
        """

        return aux_line_physics.error_propagation_voigt(FWHM_Lorentz, FWHM_Gauss, error_lorentz, error_gauss)

    def equiv_width_error(self, integrand, integration_limits, line_parameters, line_par_errors, n_samples=300, n_points=300):

        """
        Uncertainty of a modeled equivalent width, from parameter samples drawn from the asymmetric errors (split normal distributions). Kept for
        compatibility: line_physics() uses the MCMC posterior samples directly.

        Returns
        -------
        EQW_error_down, EQW_error_up: floats
        """

        rng = np.random.default_rng()
        samples = aux_line_physics.split_normal_samples(line_parameters, line_par_errors, n_samples, rng)
        lam = np.linspace(integration_limits[0], integration_limits[1], n_points)
        values = np.array([trapezoid(integrand(lam, p), lam) for p in samples])
        _, down, up = aux_line_physics.summarize(values)
        return down, up

    def _plot_line_profile(self, x, y, other_components, left, right, profile, line_name, model_curve=None):

        """Plot of a line profile with its integration limits, centroid, line dispersion and FWHM (the legend is placed outside the axes)."""

        figure, ax = plt.subplots(figsize=(12, 4.2))
        if other_components is not None:
            ax.plot(x, y + other_components, color="gray", linewidth=0.7, alpha=0.6, label="Data - continuum")
            ax.plot(x, other_components, color="gray", linestyle="--", linewidth=0.8, label="Other components of the blend")
        ax.plot(x, y, color="C1", linewidth=0.9, label="Line profile (data)")
        smoothed = profile.get("smoothed_profile")
        if smoothed is not None:
            ax.plot(x[left:right + 1], smoothed, color="C1", linewidth=1.8, alpha=0.8,
                    label=f"Smoothed profile (for the FWHM; $\\sigma$ = {profile['fwhm_smoothing_pixels']:.1f} px)")
        if model_curve is not None:
            ax.plot(*model_curve, color="k", linewidth=0.8, label="Model (median)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvspan(x[left], x[right], color="C2", alpha=0.1, label="Integration limits")
        # The measured quantities are shown in the observed frame and before the instrumental correction (if any), i.e. exactly as measured
        # on this profile:
        centroid, dispersion = profile["centroid"][0], profile.get("dispersion_observed", np.nan)
        x_left, x_right, half, peak = profile.get("fwhm_points", (np.nan,)*4)
        if np.isfinite(centroid):
            ax.axvline(centroid, color="C0", linestyle=":", label="Centroid (data)")
            if np.isfinite(dispersion):
                ax.axvspan(centroid - dispersion, centroid + dispersion, color="C0", alpha=0.15, label=r"Centroid $\pm\ \sigma_{line}$ (data)")
        if np.isfinite(x_left) and np.isfinite(x_right):
            ax.plot([x_left, x_right], [half, half], color="C3", linewidth=1.5, marker="|", ms=10, label="FWHM (data, model-independent)")
            ax.plot([x_left, x_right], [peak, peak], color="C3", linewidth=0.8, linestyle="--",
                    label="Peak (smoothed data)" if smoothed is not None else "Peak (parabola fit to the data)")
        ax.set_ylabel(r"F$_{\lambda}$ - continuum" + "\n" + r"[erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]")
        ax.set_xlabel(r"Observed $\lambda$ [$\AA$]")
        ax.set_xlim(x.min(), x.max())
        ax.grid(which="both", linestyle=":")
        ax.minorticks_on()
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False)
        if line_name is not None:
            ax.set_title(f"{line_name} - model-independent measurements (observed frame)")
        figure.tight_layout()

    def _line_posterior(self, number, par_values, par_errors, par_names, n_samples, rng):

        """
        Posterior samples of a line: those stored by fit_lines() (self.line_posteriors), if they correspond to this line; otherwise, samples
        drawn from the values and asymmetric errors (independent parameters).
        """

        stored = getattr(self, "line_posteriors", None)
        names = getattr(self, "line_names", None)
        if stored is not None and names is not None and number < len(stored) and stored[number]["line"] == names[number] \
           and np.isclose(np.median(stored[number]["samples"][:, 0]), par_values[1], rtol=0, atol=1e-6*abs(par_values[1]) + 1e-9):
            return stored[number]
        # Fallback: independent (split normal) samples from the values and errors
        names_without_z = [n for n in par_names if n != "redshift"]
        values = np.array([v for v, n in zip(par_values, par_names) if n != "redshift"], dtype=float)
        errors = np.array([e for e, n in zip(par_errors, par_names) if n != "redshift"], dtype=float)
        if "skewness" in names_without_z:
            model = "Skewedgaussian" if "fwhm_Gauss" in names_without_z else "Skewedlorentzian"
        elif "fwhm_Gauss" in names_without_z and "fwhm_Lorentz" in names_without_z:
            model = "Voigt"
        elif "fwhm_Lorentz" in names_without_z:
            model = "Lorentz"
        else:
            model = "Gaussian"
        samples = aux_line_physics.split_normal_samples(values, errors, n_samples, rng)
        z_index = par_names.index("redshift") if "redshift" in par_names else None
        z_samples = aux_line_physics.split_normal_samples([par_values[z_index]], [par_errors[z_index]], n_samples, rng)[:, 0] \
                    if z_index is not None else np.zeros(n_samples)
        return {"line": None, "model": model, "labels": names_without_z, "samples": samples, "z_samples": z_samples, "group": None,
                "group_models": None, "group_samples": None, "error_scale": 1.0}

    @_plot_style
    def line_physics(self, wavelengths, flux_density, continuum_baseline, par_values_list, par_values_errors_list, par_names_list, line_windows,
                     line_std_deviation, sistemic_redshift = None, plot = True, save_file = True, instrumental_fwhm = "auto", n_realizations = 500,
                     systemic_redshift = None):

        """
        In this function, we compute the physical properties of the fitted lines: flux, equivalent width, FWHM, line dispersion and dispersion
        velocity, each one in two ways:

        - Modeled (from the line model): computed for each sample of the MCMC posterior distribution of the line parameters (stored by
          fit_lines()), so that the uncertainties include the correlations between the parameters (e.g. amplitude and width) and the
          uncertainty of the redshift. The flux is the integral of the model (analytic for the Gaussian, Lorentzian and Voigt profiles). For a
          blended line, only its own component is used, so the modeled quantities are the best way to measure blended lines.
        - Profile (model-independent): measured directly on the continuum-subtracted data, between the points where the line profile reaches
          the noise level, with Monte Carlo uncertainties from the uncertainty of each pixel. For a blended line, the other components of the
          blend (median models) are subtracted first.

        The equivalent width is the integral of (Fc - F)/Fc, i.e. positive for absorption lines and negative for emission lines, and the
        integrated flux is positive for emission lines. Rest-frame quantities are the observed ones divided by (1 + z).

        Notes
        -----
        - The second moment of profiles with Lorentzian wings (Lorentzian, Voigt and skewed Lorentzian) diverges, so their modeled line
          dispersion (and dispersion velocity) is computed within the integration limits of the profile measurement: it depends on these
          limits, but is directly comparable to the profile (model-independent) dispersion. For Gaussian models it is the analytic value.
        - The profile FWHM does not use the model. For noisy lines (peak S/N < 15), it is measured on the profile smoothed with a Gaussian
          kernel (narrower than ~1/3 of the line width), whose broadening is then removed in quadrature; its Monte Carlo uncertainty is
          large for weak lines, for which the modeled FWHM is usually the better measurement.
        - The uncertainty of the continuum level is not included.
        - If the posterior samples of fit_lines() are not available (e.g. the parameters were edited or loaded from a file), the parameters
          are sampled from their values and asymmetric errors, as if they were independent.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        continuum_baseline: numpy.ndarray (float)
            The continuum flux density (output of analysis.find_lines()).
        par_values_list, par_values_errors_list, par_names_list: lists
            The best-fit values, errors and parameter names of each line (outputs of analysis.fit_lines()).
        line_windows: list
            The wavelength window of each line (output of analysis.fit_lines()).
        line_std_deviation: numpy.ndarray (float)
            The noise of the local continuum at each line (output of analysis.find_lines()); used when the spectrum has no flux uncertainties.
        systemic_redshift: float or [float, float]
            Optional. The systemic redshift (and, optionally, its uncertainty) used to convert all quantities to the rest frame. By default, the
            redshift of each line is used. ('sistemic_redshift' is accepted as an alias, for compatibility.)
        plot: boolean
            If True, the profile of each line is shown with its integration limits, centroid, line dispersion and FWHM.
        save_file: boolean
            If True, the results are saved in "<target>_line_physics.csv" in the output directory.
        instrumental_fwhm: "auto", float or None
            The instrumental resolution (FWHM, in Angstroms, in the observed frame), removed from the FWHMs and line dispersions, which is
            important for lines that are not much broader than the resolution. "auto" (default): the resolution stored in the spectrum file
            (measured from the width of the lamp lines in the wavelength calibration, see load_calibrated_data()), at the wavelength of each
            line; if the file does not have it, no correction is applied (with a note). A number: the same FWHM for all lines (e.g. the FWHM
            of the lamp lines, or of the sky lines). None: no correction. The correction depends on the model: in quadrature for Gaussian
            profiles, by inverting the Voigt FWHM for Lorentzian profiles, and on the Gaussian component for Voigt profiles.
        n_realizations: int
            Number of Monte Carlo realizations for the profile measurements. Default is 500.

        Returns
        -------
        profile_equiv_width_rest_frame, modeled_equiv_width_rest_frame: Quantity (Angstrom)
            For each line, [value, error_down, error_up] of the rest-frame equivalent width.
        profile_integrated_flux, modeled_integrated_flux: Quantity (erg/cm2/s)
            The line fluxes.
        profile_rest_frame_disp_velocity, modeled_rest_frame_disp_velocity: Quantity (km/s)
            The dispersion velocities, c*sigma/lambda.
        profile_line_dispersion_rest_frame: Quantity (Angstrom)
            The rest-frame line dispersion (second moment of the profile).
        profile_rest_frame_fwhm, modeled_rest_frame_fhwm: Quantity (Angstrom)
            The rest-frame FWHMs.

        All the measurements (including the modeled line dispersion, the FWHM in km/s and the centroid of the profile) are also stored in
        self.line_physics_results (a list of dictionaries).
        """

        if systemic_redshift is None:
            systemic_redshift = sistemic_redshift
        x_all = self._values(wavelengths)
        f_all = self._values(flux_density)
        continuum_all = np.asarray(continuum_baseline, dtype=float)
        good = self._good_pixels(x_all, f_all)
        flux_err = self._matching(self.flux_err, len(f_all))
        noise_at_lines = np.atleast_1d(np.asarray(getattr(line_std_deviation, "value", line_std_deviation), dtype=float))
        n_lines = len(par_values_list)
        if not (len(par_values_errors_list) == len(par_names_list) == len(line_windows) == n_lines):
            raise ValueError("par_values_list, par_values_errors_list, par_names_list and line_windows must have one entry per line.")
        line_names = list(getattr(self, "line_names", None) or [f"line_{i}" for i in range(n_lines)])
        if len(line_names) != n_lines:
            line_names = [f"line_{i}" for i in range(n_lines)]
        rng = np.random.default_rng()

        def continuum_function(wavelength):
            return np.interp(wavelength, x_all, continuum_all)

        # Instrumental resolution at each line:
        if isinstance(instrumental_fwhm, str):
            if instrumental_fwhm.lower() != "auto":
                raise ValueError("instrumental_fwhm must be 'auto', a number (FWHM in Angstrom) or None.")
            stored = getattr(self, "instrumental_fwhm", None)
            stored = self._matching(stored, len(x_all)) if stored is not None else None
            if stored is None:
                print("Note: the spectrum file has no instrumental resolution, so the line widths are not corrected for it. Give "
                      "instrumental_fwhm=<FWHM in Angstrom> (e.g. the FWHM of the lamp lines) to correct them.")
                resolution_at = lambda wavelength: None
            else:
                valid = np.isfinite(stored) & (stored > 0)
                resolution_at = lambda wavelength: float(np.interp(wavelength, x_all[valid], stored[valid]))
        elif instrumental_fwhm is None:
            resolution_at = lambda wavelength: None
        else:
            constant = float(getattr(instrumental_fwhm, "value", instrumental_fwhm))
            if not constant >= 0:
                raise ValueError("instrumental_fwhm must be a non-negative number (FWHM in Angstrom).")
            resolution_at = lambda wavelength: constant

        results = []
        for number in range(n_lines):
            values, errors, names = par_values_list[number], par_values_errors_list[number], par_names_list[number]
            posterior = self._line_posterior(number, values, errors, names, 4000, rng)
            samples = posterior["samples"]
            if systemic_redshift is not None:
                z_value, z_error = (systemic_redshift if np.ndim(systemic_redshift) else (systemic_redshift, 0.0))
                z_samples = z_value + float(z_error)*rng.standard_normal(len(samples))
            else:
                z_samples = posterior["z_samples"]

            # ------------------------------------------------------------ Profile (model-independent) quantities
            window = line_windows[number]
            inside = (x_all > window[0]) & (x_all < window[1]) & good
            x, y = x_all[inside], (f_all - continuum_all)[inside]
            continuum = continuum_all[inside]
            error_scale = posterior.get("error_scale", 1.0)
            error = (flux_err[inside]*error_scale if flux_err is not None else np.full(len(x), noise_at_lines[min(number, len(noise_at_lines) - 1)]))
            other = None
            if posterior.get("group") and len(posterior["group"]) > 1:
                # Subtract the median model of the other components of the blend
                other = np.zeros(len(x))
                draw = rng.choice(len(samples), size=min(200, len(samples)), replace=False)
                for name, model_name, component_samples in zip(posterior["group"], posterior["group_models"], posterior["group_samples"]):
                    if name == posterior["line"]:
                        continue
                    model = aux_line_physics.MODEL_FUNCTIONS[model_name]
                    other += np.median([model(theta, x) for theta in component_samples[draw]], axis=0)
                y = y - other
            location = float(np.median(samples[:, 0]))
            line_resolution = resolution_at(location)
            peak = extraction.find_nearest(x, location)
            left, right = aux_line_physics.integration_limits(x, y, error, peak, x.min(), x.max())
            self.integration_limits_dispersion = [x[left], x[right]]
            profile = aux_line_physics.profile_physics(x[left:right + 1], y[left:right + 1], error[left:right + 1], continuum[left:right + 1],
                                                       z_samples, n_realizations=n_realizations, instrumental_fwhm=line_resolution, rng=rng)
            if left == 0 or right == len(x) - 1:
                warnings.warn(f"{line_names[number]}: the line profile reaches the edge of its window ({window[0]:.0f}-{window[1]:.0f} A) above the noise "
                              "level, so the profile (model-independent) flux and equivalent width miss part of the wings (typical of lines with "
                              "Lorentzian wings). Prefer the modeled values, or fit the line with a larger window.")
            if right - left < 4:
                warnings.warn(f"{line_names[number]}: the line profile spans only {right - left + 1} pixels above the noise; the model-independent "
                              "measurements are unreliable.")
            # ------------------------------------------------------------ Modeled quantities (posterior samples)
            # (the line dispersion of models with Lorentzian wings, which diverges, is computed within the integration limits of the profile)
            modeled = aux_line_physics.model_physics(posterior["model"], samples, z_samples, continuum_function, instrumental_fwhm=line_resolution,
                                                     rng=rng, dispersion_limits=(float(x[left]), float(x[right])))

            results.append({"line": line_names[number], "model": posterior["model"], "modeled": modeled, "profile": profile,
                            "integration_limits": [float(x[left]), float(x[right])], "instrumental_fwhm": line_resolution,
                            "posterior": "MCMC samples" if posterior["line"] is not None else "independent errors"})

            if plot:
                model = aux_line_physics.MODEL_FUNCTIONS[posterior["model"]]
                grid = np.linspace(x.min(), x.max(), 1000)
                curve = (grid, model(np.median(samples, axis=0), grid))
                self._plot_line_profile(x, y, other, left, right, profile, line_names[number], model_curve=curve)

        # ------------------------------------------------------------ Outputs (same structure as previous versions)
        def collect(kind, key):
            return [result[kind][key] for result in results]
        profile_equiv_width_rest_frame = collect("profile", "equivalent_width_rest")*u.AA
        modeled_equiv_width_rest_frame = collect("modeled", "equivalent_width_rest")*u.AA
        profile_integrated_flux = collect("profile", "flux")*u.erg/u.cm**2/u.s
        modeled_integrated_flux = collect("modeled", "flux")*u.erg/u.cm**2/u.s
        profile_rest_frame_disp_velocity = collect("profile", "dispersion_velocity")*u.km/u.s
        modeled_rest_frame_disp_velocity = collect("modeled", "dispersion_velocity")*u.km/u.s
        profile_line_dispersion_rest_frame = collect("profile", "dispersion_rest")*u.AA
        profile_rest_frame_fwhm = collect("profile", "fwhm_rest")*u.AA
        modeled_rest_frame_fhwm = collect("modeled", "fwhm_rest")*u.AA
        self.line_physics_results = results

        for result in results:
            m, p = result["modeled"], result["profile"]
            print(f"{result['line']} ({result['model']}): flux = {m['flux'][0]:.3e} (model) / {p['flux'][0]:.3e} (profile) erg/cm2/s; "
                  f"EW_rest = {m['equivalent_width_rest'][0]:.2f} / {p['equivalent_width_rest'][0]:.2f} A; "
                  f"FWHM_rest = {m['fwhm_rest'][0]:.2f} / {p['fwhm_rest'][0]:.2f} A ({m['fwhm_velocity'][0]:.0f} km/s, model)"
                  + (f"; corrected for an instrumental FWHM of {result['instrumental_fwhm']:.2f} A" if result["instrumental_fwhm"] else ""))
        if any(r["posterior"] != "MCMC samples" for r in results):
            print("Note: for some lines the MCMC samples of fit_lines() were not available, so their parameters were sampled from the values "
                  "and errors as independent variables.")

        if plot:
            plt.show()

        if save_file:
            columns = [("profile_equiv_width_rest_frame [Angstrom]", profile_equiv_width_rest_frame),
                       ("modeled_equiv_width_rest_frame [Angstrom]", modeled_equiv_width_rest_frame),
                       ("profile_integrated_flux [erg/cm2/s]", profile_integrated_flux),
                       ("modeled_integrated_flux [erg/cm2/s]", modeled_integrated_flux),
                       ("profile_rest_frame_disp_velocity [km/s]", profile_rest_frame_disp_velocity),
                       ("modeled_rest_frame_disp_velocity [km/s]", modeled_rest_frame_disp_velocity),
                       ("profile_line_dispersion_rest_frame [Angstrom]", profile_line_dispersion_rest_frame),
                       ("profile_rest_frame_fwhm [Angstrom]", profile_rest_frame_fwhm),
                       ("modeled_rest_frame_fhwm [Angstrom]", modeled_rest_frame_fhwm)]
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.output_dir) / f"{self.target_name}_line_physics.csv", "w") as output:
                output.write("Line name, " + ", ".join(f"{name}, err_down, err_up" for name, _ in columns) + "\n")
                for number, name in enumerate(line_names):
                    output.write(name + "".join(f", {value:.6g}" for _, quantity in columns for value in quantity[number].value) + "\n")

        return profile_equiv_width_rest_frame, modeled_equiv_width_rest_frame, profile_integrated_flux, modeled_integrated_flux, profile_rest_frame_disp_velocity, modeled_rest_frame_disp_velocity, profile_line_dispersion_rest_frame, profile_rest_frame_fwhm, modeled_rest_frame_fhwm



    # ==================================================================================================================================
    # Black hole masses (single-epoch virial estimators)
    # ==================================================================================================================================

    def _virial_mass(self, label, terms, zero_point, intrinsic_scatter, n_samples=20000):

        """
        log10(M_BH/M_sun) = zero_point + sum(slope*log10(quantity/normalization)) for samples of the quantities, and its uncertainty: the
        statistical part (16%-84% range of the samples, from the measurement errors) added in quadrature to the intrinsic scatter of the
        calibration, which dominates. Returns [log10 M, total uncertainty] and stores the details in self.BH_mass_results[label].
        """

        log_mass = zero_point + sum(slope*np.log10(samples/normalization) for samples, normalization, slope in terms)
        log_mass = log_mass[np.isfinite(log_mass)]
        if len(log_mass) == 0:
            warnings.warn(f"{label}: the black hole mass could not be computed (non-positive FWHM, flux or luminosity).", stacklevel=3)
            return [np.nan, np.nan]
        low, median, high = np.quantile(log_mass, [0.16, 0.5, 0.84])
        statistical = 0.5*(high - low)
        total = float(np.sqrt(statistical**2 + intrinsic_scatter**2))
        if not hasattr(self, "BH_mass_results"):
            self.BH_mass_results = {}
        self.BH_mass_results[label] = {"log10_mass": float(median), "statistical_error": float(statistical),
                                       "intrinsic_scatter": intrinsic_scatter, "total_error": total}
        print(f"{label}: log10(M_BH/M_sun) = {median:.2f} +- {statistical:.2f} (measurement) +- {intrinsic_scatter:.2f} (calibration) dex")
        return [float(median), total]

    def _mass_inputs(self, fwhm, rest_wavelength, systemic_redshift, H0, Om0, n_samples, rng):

        """Samples of the FWHM (km/s), of the redshift and of the luminosity distance."""

        fwhm_value, fwhm_down, fwhm_up = aux_line_physics.value_and_errors(fwhm, u.AA)
        z_value, z_down, z_up = aux_line_physics.value_and_errors(systemic_redshift)
        fwhm_kms = aux_line_physics.SPEED_OF_LIGHT*aux_line_physics.asymmetric_samples(fwhm_value, fwhm_down, fwhm_up, n_samples, rng,
                                                                                       positive=True)/rest_wavelength
        z_samples = aux_line_physics.asymmetric_samples(z_value, z_down, z_up, n_samples, rng)
        if fwhm_kms is not None and np.nanmedian(fwhm_kms) < 1000:
            warnings.warn(f"FWHM = {np.nanmedian(fwhm_kms):.0f} km/s: the virial estimators are calibrated for broad lines (FWHM > ~1000-2000 "
                          "km/s). Check that the FWHM is in the rest frame (Angstrom) and corrected for the instrumental resolution.", stacklevel=3)
        distance = aux_line_physics.luminosity_distance_cm(z_samples, H0, Om0)
        return fwhm_kms, z_value, z_samples, distance

    def _continuum_luminosity(self, wavelengths, continuum_baseline, z_value, z_samples, distance, rest_wavelengths):

        """
        Samples of the monochromatic continuum luminosity lambda*L_lambda (erg/s) at the first rest wavelength of 'rest_wavelengths' covered by
        the spectrum (the continuum is averaged over +-25 rest-frame Angstroms). Returns (samples, rest wavelength used) or (None, None).
        """

        x = self._values(wavelengths)
        continuum = np.asarray(getattr(continuum_baseline, "value", continuum_baseline), dtype=float)
        for rest_wavelength in rest_wavelengths:
            flux_density, observed = aux_line_physics.continuum_flux_density(x, continuum, z_value, rest_wavelength)
            if np.isfinite(flux_density) and flux_density > 0:
                return observed*flux_density*4*np.pi*distance**2, rest_wavelength   # lambda_obs*F_lambda,obs = lambda_rest*F_lambda,rest
        coverage = f"{x.min()/(1 + z_value):.0f}-{x.max()/(1 + z_value):.0f} A (rest frame)"
        warnings.warn(f"The continuum at {', '.join(f'{w:.0f}' for w in rest_wavelengths)} A (rest frame) is not covered by the spectrum "
                      f"({coverage}), so the luminosity-based mass cannot be computed.", stacklevel=3)
        return None, None

    def BH_mass_Hbeta_VP2006(self, wavelengths, continuum_baseline, FWHM_Hbeta, sistemic_redshift, integrated_flux_Hbeta, Hbeta_rest_wavelength = 4861.333,
                             H0 = 70, Om0 = 0.3, n_samples = 20000):

        """
        Black hole mass from the H-beta line, with the calibrations of Vestergaard & Peterson 2006, ApJ, 641, 689 (their Eqs. 5 and 6), in a
        flat cosmology with H0 = 70 km/s/Mpc and Omega_matter = 0.3 (H0 and Omega_matter can be changed):

            log M = log[(FWHM/1000 km/s)^2 (lambda L_5100/1e44 erg/s)^0.50] + 6.91        (Eq. 5)
            log M = log[(FWHM/1000 km/s)^2 (L_Hbeta/1e42 erg/s)^0.63] + 6.67              (Eq. 6)

        The uncertainty is the intrinsic scatter of the calibration (0.43 dex) added in quadrature to the propagated measurement errors of
        the FWHM, of the flux and of the redshift (if given).

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            The continuum flux density (output of analysis.find_lines()), averaged over 5075-5125 A in the rest frame.
        FWHM_Hbeta: float, Quantity or [value, error_down, error_up]
            The rest-frame FWHM of the broad H-beta line in Angstroms, corrected for the instrumental resolution, e.g. one row of the
            output modeled_rest_frame_fhwm of line_physics().
        sistemic_redshift: float or [value, error]
            The systemic redshift of the AGN.
        integrated_flux_Hbeta: float, Quantity or [value, error_down, error_up]
            The flux of the broad H-beta line in erg/cm2/s.
        Hbeta_rest_wavelength: float
            The rest wavelength of the H-beta line in Angstroms.
        H0, Om0: floats
            The cosmological parameters. Defaults: 70 km/s/Mpc and 0.3.
        n_samples: int
            Number of Monte Carlo samples for the propagation of the measurement errors.

        Returns
        -------
        log10_BH_mass_continuum: list
            [log10(M_BH/M_sun), uncertainty in dex] from Eq. 5 ([nan, nan] if the continuum at 5100 A is not covered by the spectrum).
        log10_BH_mass_line_lum: list
            [log10(M_BH/M_sun), uncertainty in dex] from Eq. 6.
        """

        rng = np.random.default_rng()
        fwhm, z_value, z_samples, distance = self._mass_inputs(FWHM_Hbeta, Hbeta_rest_wavelength, sistemic_redshift, H0, Om0, n_samples, rng)
        flux_value, flux_down, flux_up = aux_line_physics.value_and_errors(integrated_flux_Hbeta, u.erg/u.cm**2/u.s)
        line_luminosity = aux_line_physics.asymmetric_samples(flux_value, flux_down, flux_up, n_samples, rng, positive=True)*4*np.pi*distance**2
        luminosity, _ = self._continuum_luminosity(wavelengths, continuum_baseline, z_value, z_samples, distance, [5100])
        mass_continuum = [np.nan, np.nan] if luminosity is None else \
            self._virial_mass("Hbeta (VP06, L5100)", [(fwhm, 1000, 2), (luminosity, 1e44, 0.5)], 6.91, 0.43)
        mass_line = self._virial_mass("Hbeta (VP06, L_Hbeta)", [(fwhm, 1000, 2), (line_luminosity, 1e42, 0.63)], 6.67, 0.43)
        return mass_continuum, mass_line

    def BH_mass_CIV_VP2006(self, wavelengths, continuum_baseline, FWHM_CIV, sistemic_redshift, CIV_rest_wavelength = 1549.4795, H0 = 70, Om0 = 0.3,
                           n_samples = 20000):

        """
        Black hole mass from the C IV line, with the calibration of Vestergaard & Peterson 2006, ApJ, 641, 689 (their Eq. 7), in a flat
        cosmology with H0 = 70 km/s/Mpc and Omega_matter = 0.3:

            log M = log[(FWHM/1000 km/s)^2 (lambda L_1350/1e44 erg/s)^0.53] + 6.66

        The continuum luminosity at 1350 A is used (or at 1450 A, if 1350 A is not covered by the spectrum; VP06 show that both can be used).
        The uncertainty is the intrinsic scatter of the calibration (0.36 dex) added in quadrature to the propagated measurement errors.
        Note that C IV is affected by non-virial (outflow) components; see BH_mass_CIV_Coatman2017() for a correction based on the C IV blueshift.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            The continuum flux density (output of analysis.find_lines()).
        FWHM_CIV: float, Quantity or [value, error_down, error_up]
            The rest-frame FWHM of the C IV line in Angstroms, corrected for the instrumental resolution.
        sistemic_redshift: float or [value, error]
            The systemic redshift of the AGN.
        CIV_rest_wavelength: float
            The rest wavelength of the C IV line in Angstroms.
        H0, Om0: floats
            The cosmological parameters.
        n_samples: int
            Number of Monte Carlo samples for the propagation of the measurement errors.

        Returns
        -------
        log10_BH_mass_CIV: list
            [log10(M_BH/M_sun), uncertainty in dex].
        """

        rng = np.random.default_rng()
        fwhm, z_value, z_samples, distance = self._mass_inputs(FWHM_CIV, CIV_rest_wavelength, sistemic_redshift, H0, Om0, n_samples, rng)
        luminosity, used = self._continuum_luminosity(wavelengths, continuum_baseline, z_value, z_samples, distance, [1350, 1450])
        if luminosity is None:
            return [np.nan, np.nan]
        return self._virial_mass(f"CIV (VP06, L{used})", [(fwhm, 1000, 2), (luminosity, 1e44, 0.53)], 6.66, 0.36)

    def BH_mass_CIV_Coatman2017(self, wavelengths, continuum_baseline, FWHM_CIV, CIV_redshift, sistemic_redshift, CIV_rest_wavelength = 1549.48, H0 = 71,
                                Om0 = 0.27, blueshift = None, n_samples = 20000):

        """
        Black hole mass from the C IV line, corrected for the non-virial contribution of outflows with the C IV blueshift (Coatman et al.
        2017, MNRAS, 465, 2120), in a flat cosmology with H0 = 71 km/s/Mpc and Omega_matter = 0.27:

            FWHM_corrected = FWHM/(0.41*blueshift/(1000 km/s) + 0.62)
            log M = log[(FWHM_corrected/1000 km/s)^2 (lambda L_1350/1e44 erg/s)^0.53] + 6.71

        The blueshift is c*(1549.48 - lambda_half)/1549.48, where lambda_half is the wavelength that bisects the C IV flux, in the rest frame of
        the systemic redshift; it is positive when C IV is blueshifted. Here it is computed from the redshift of C IV (which should therefore
        be measured from the line centroid, e.g. the profile centroid of line_physics()) as c*(z_systemic - z_CIV)/(1 + z_systemic), or it
        can be given directly with 'blueshift'. The correction was calibrated for blueshifts between ~0 and ~5000 km/s. The uncertainty is
        0.36 dex (the scatter of VP06; Coatman et al. do not give it separately) added in quadrature to the measurement errors.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            The continuum flux density (output of analysis.find_lines()).
        FWHM_CIV: float, Quantity or [value, error_down, error_up]
            The rest-frame FWHM of the C IV line in Angstroms, corrected for the instrumental resolution.
        CIV_redshift: float or [value, error]
            The redshift of the C IV line (from its centroid). Ignored if 'blueshift' is given.
        sistemic_redshift: float or [value, error]
            The systemic redshift of the AGN (e.g. from narrow lines, or from low-ionization lines such as Mg II).
        CIV_rest_wavelength: float
            The rest wavelength of the C IV line in Angstroms (vacuum).
        H0, Om0: floats
            The cosmological parameters. Defaults: 71 km/s/Mpc and 0.27, as in Coatman et al. 2017.
        blueshift: float or [value, error]
            Optional. The C IV blueshift in km/s (positive for blueshifted lines).
        n_samples: int
            Number of Monte Carlo samples for the propagation of the measurement errors.

        Returns
        -------
        log10_BH_mass_CIV: list
            [log10(M_BH/M_sun), uncertainty in dex].
        """

        rng = np.random.default_rng()
        fwhm, z_value, z_samples, distance = self._mass_inputs(FWHM_CIV, CIV_rest_wavelength, sistemic_redshift, H0, Om0, n_samples, rng)
        if blueshift is not None:
            value, down, up = aux_line_physics.value_and_errors(blueshift)
            blueshift_samples = aux_line_physics.asymmetric_samples(value, down, up, n_samples, rng)
        else:
            value, down, up = aux_line_physics.value_and_errors(CIV_redshift)
            civ_samples = aux_line_physics.asymmetric_samples(value, down, up, n_samples, rng)
            blueshift_samples = aux_line_physics.SPEED_OF_LIGHT*(z_samples - civ_samples)/(1 + z_samples)
        median_blueshift = float(np.nanmedian(blueshift_samples))
        print(f"C IV blueshift = {median_blueshift:.0f} km/s")
        if not -500 < median_blueshift < 5500:
            warnings.warn(f"C IV blueshift = {median_blueshift:.0f} km/s is outside the range where the correction of Coatman et al. (2017) was "
                          "calibrated (~0-5000 km/s). Check the redshifts (C IV from its centroid, and the systemic one).", stacklevel=2)
        denominator = 0.41*blueshift_samples/1000 + 0.62
        if np.mean(denominator > 0) < 0.5:
            warnings.warn(f"The correction of Coatman et al. (2017), FWHM/(0.41*blueshift/1000 + 0.62), is not defined for C IV blueshifts below "
                          f"-1512 km/s (C IV redshifted with respect to the systemic redshift), and the blueshift here is {median_blueshift:.0f} km/s. "
                          "The mass is not computed: check the systemic redshift (preferably from a low-ionization line, such as Mg II or Hbeta, or "
                          "from narrow lines), or use BH_mass_CIV_VP2006().", stacklevel=2)
            return [np.nan, np.nan]
        corrected_fwhm = np.where(denominator > 0, fwhm/denominator, np.nan)
        luminosity, used = self._continuum_luminosity(wavelengths, continuum_baseline, z_value, z_samples, distance, [1350, 1450])
        if luminosity is None:
            return [np.nan, np.nan]
        return self._virial_mass(f"CIV (Coatman+17, L{used})", [(corrected_fwhm, 1000, 2), (luminosity, 1e44, 0.53)], 6.71, 0.36)


    def BH_mass_MgII_VO2009(self, wavelengths, continuum_baseline, FWHM_MgII, sistemic_redshift, MgII_rest_wavelength = 2798.75, H0 = 70, Om0 = 0.3,
                            n_samples = 20000):

        """
        Black hole mass from the Mg II line, with the calibrations of Vestergaard & Osmer 2009, ApJ, 699, 800 (their Eq. 1), in a flat
        cosmology with H0 = 70 km/s/Mpc and Omega_matter = 0.3:

            log M = log[(FWHM/1000 km/s)^2 (lambda L_lambda/1e44 erg/s)^0.5] + zero point,

        with zero point = 6.86 for the continuum luminosity at 3000 A, or 6.79 at 2100 A (used if 3000 A is not covered by the spectrum). The uncertainty is the intrinsic scatter of the calibration (0.55 dex) added in
        quadrature to the propagated measurement errors.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            The continuum flux density (output of analysis.find_lines()).
        FWHM_MgII: float, Quantity or [value, error_down, error_up]
            The rest-frame FWHM of the Mg II line in Angstroms, corrected for the instrumental resolution.
        sistemic_redshift: float or [value, error]
            The systemic redshift of the AGN.
        MgII_rest_wavelength: float
            The rest wavelength of the Mg II line in Angstroms.
        H0, Om0: floats
            The cosmological parameters.
        n_samples: int
            Number of Monte Carlo samples for the propagation of the measurement errors.

        Returns
        -------
        log10_BH_mass_MgII: list
            [log10(M_BH/M_sun), uncertainty in dex].
        """

        rng = np.random.default_rng()
        zero_points = {3000: 6.86, 2100: 6.79}
        fwhm, z_value, z_samples, distance = self._mass_inputs(FWHM_MgII, MgII_rest_wavelength, sistemic_redshift, H0, Om0, n_samples, rng)
        luminosity, used = self._continuum_luminosity(wavelengths, continuum_baseline, z_value, z_samples, distance, list(zero_points))
        if luminosity is None:
            return [np.nan, np.nan]
        return self._virial_mass(f"MgII (VO09, L{used})", [(fwhm, 1000, 2), (luminosity, 1e44, 0.5)], zero_points[used], 0.55)

    def BH_mass_Halpha_Shen2011(self, FWHM_Halpha, sistemic_redshift, integrated_flux_Halpha, Halpha_rest_wavelength = 6562.819, H0 = 70, Om0 = 0.3,
                                systematic_error = 0.18, n_samples = 20000):

        """
        Black hole mass from the broad H-alpha line, with the calibration used by Shen et al. 2011, ApJS, 194, 45 (their Eq. 10, based on
        Greene & Ho 2005), in a flat cosmology with H0 = 70 km/s/Mpc and Omega_matter = 0.3:

            log M = 0.379 + 0.43 log(L_Halpha/1e42 erg/s) + 2.1 log(FWHM/km/s)

        Parameters
        ----------
        FWHM_Halpha: float, Quantity or [value, error_down, error_up]
            The rest-frame FWHM of the broad H-alpha line in Angstroms, corrected for the instrumental resolution.
        sistemic_redshift: float or [value, error]
            The systemic redshift of the AGN.
        integrated_flux_Halpha: float, Quantity or [value, error_down, error_up]
            The flux of the broad H-alpha line in erg/cm2/s.
        Halpha_rest_wavelength: float
            The rest wavelength of the H-alpha line in Angstroms.
        H0, Om0: floats
            The cosmological parameters.
        systematic_error: float
            The calibration uncertainty in dex, added in quadrature to the measurement errors. The default, 0.18 dex, is the scatter between the
            H-alpha and H-beta based masses reported by Shen et al. 2011; the absolute uncertainty of single-epoch virial masses is larger
            (~0.4 dex), as for the other estimators.
        n_samples: int
            Number of Monte Carlo samples for the propagation of the measurement errors.

        Returns
        -------
        log10_BH_mass_Halpha: list
            [log10(M_BH/M_sun), uncertainty in dex].
        """

        rng = np.random.default_rng()
        fwhm, z_value, z_samples, distance = self._mass_inputs(FWHM_Halpha, Halpha_rest_wavelength, sistemic_redshift, H0, Om0, n_samples, rng)
        flux_value, flux_down, flux_up = aux_line_physics.value_and_errors(integrated_flux_Halpha, u.erg/u.cm**2/u.s)
        luminosity = aux_line_physics.asymmetric_samples(flux_value, flux_down, flux_up, n_samples, rng, positive=True)*4*np.pi*distance**2
        return self._virial_mass("Halpha (Shen+11)", [(fwhm, 1, 2.1), (luminosity, 1e42, 0.43)], 0.379, systematic_error)


    def get_highest_resolution(self, specs_dict):

        """
        This function gets the spectrum with the finest wavelength sampling in specs_dict.

        Parameters
        -----------
        specs_dict: dictionary
            A dictionary where each key is a spectrum name and each value is a list containing [lambda0, lambdaf, R], with R the
            number of pixels per Angstrom (the sampling, which sets the resolution of the stacked grid).

        Returns
        --------
        highest_R_spec: string
            The name of the spectrum with the finest sampling.
        highest_R: float
            The highest value of R found in specs_dict.
        """

        if len(specs_dict) == 0:
            raise ValueError("specs_dict is empty.")
        highest_R_spec = max(specs_dict, key=lambda name: specs_dict[name][2])
        return highest_R_spec, specs_dict[highest_R_spec][2]


    def create_common_grid(self, specs_dict, over_resolution_factor=1):

        """
        This function creates a common (linear) wavelength grid for all spectra, to be used for the stacked spectrum. It covers from the
        smallest lambda0 to the largest lambdaf, with a step equal to 1/(highest R * over_resolution_factor).

        Parameters
        -----------
        specs_dict: dictionary
            A dictionary where each key is a spectrum and each value is a list containing [lambda0, lambdaf, R] (R in pixels per Angstrom).
        over_resolution_factor: float
            The grid step is the finest pixel size divided by this factor. Default: 1 (the native sampling of the best-sampled spectrum).

        Returns
        --------
        wavelengths_stacked: numpy.ndarray
            An array containing the wavelength values of the common grid.
        """

        if over_resolution_factor <= 0:
            raise ValueError("'over_resolution_factor' must be positive.")
        highest_R = self.get_highest_resolution(specs_dict)[1]
        lower = min(value[0] for value in specs_dict.values())
        upper = max(value[1] for value in specs_dict.values())
        step = 1.0/(highest_R*over_resolution_factor)
        return lower + step*np.arange(int(np.floor((upper - lower)/step)) + 1)


    def interpolate_spectra(self, list_of_wavelengths, list_of_fluxes, wavelengths_stacked, list_of_errors=None, list_of_masks=None):

        """
        This function resamples all spectra onto the common wavelength grid of the stacked spectrum, conserving the flux (each new pixel
        is the overlap-weighted average of the old pixels; no spline ringing around strong lines or gaps), and creates a matrix where
        each row is a resampled spectrum, with NaN where there is no (good) data.

        Parameters
        -----------
        list_of_wavelengths: list
            A list where each element is a numpy array containing the wavelength values of each spectrum.
        list_of_fluxes: list
            A list where each element is a numpy array containing the flux values of each spectrum.
        wavelengths_stacked: numpy.ndarray
            The wavelength values of the common grid.
        list_of_errors: list
            Optional. The 1-sigma flux uncertainties of each spectrum.
        list_of_masks: list
            Optional. The data quality masks of each spectrum (pixels with the bits 1, 4 or 8 are excluded).

        Returns
        --------
        spectra_interp_matrix: numpy.ndarray
            A matrix where each row is a resampled spectrum with the size of the common wavelength grid. If list_of_errors is given,
            the matrix of the resampled uncertainties is also returned.
        """

        n_spectra = len(list_of_wavelengths)
        errors = list_of_errors if list_of_errors is not None else [None]*n_spectra
        masks = list_of_masks if list_of_masks is not None else [None]*n_spectra
        fluxes, sigmas = [], []
        for wavelength, flux, error, mask in zip(list_of_wavelengths, list_of_fluxes, errors, masks):
            new_flux, new_error, _, _ = aux_stacking.resample_spectrum(getattr(wavelength, "value", wavelength), getattr(flux, "value", flux),
                                                                    wavelengths_stacked, getattr(error, "value", error), mask=mask)
            fluxes.append(new_flux)
            sigmas.append(new_error)
        if list_of_errors is not None:
            return np.vstack(fluxes), np.vstack(sigmas)
        return np.vstack(fluxes)


    def calculate_stack(self, spectra_interp_matrix, method="median", errors_matrix=None, clip_sigma=3.0):

        """
        This function calculates the stacked spectrum pixel by pixel, ignoring NaN values.

        Parameters
        -----------
        spectra_interp_matrix: numpy.ndarray
            A matrix where each row is a resampled spectrum with the size of the common wavelength grid.
        method: string
            "median", "mean", "weighted" (inverse-variance weighted mean; needs errors_matrix) or "sigma_clip" (mean, weighted if
            errors_matrix is given, after rejecting points more than clip_sigma away from the median).
        errors_matrix: numpy.ndarray
            Optional. The 1-sigma uncertainties of spectra_interp_matrix.
        clip_sigma: float
            The rejection threshold of method="sigma_clip".

        Returns
        --------
        stacked_flux: numpy.ndarray
            The stacked flux. If errors_matrix is given, its propagated 1-sigma uncertainty is also returned.
        """

        flux, error, _, _, _ = aux_stacking.combine(spectra_interp_matrix, errors_matrix, method=method, clip_sigma=clip_sigma)
        return (flux, error) if errors_matrix is not None else flux


    def _stack_inputs(self, input_data, output_dir):

        """Read the spectra to be stacked: returns a list of dicts with name, wavelength, flux, flux_err, calibration_err, mask, wavelength_err."""

        if isinstance(input_data, (str, Path)) and Path(input_data).is_dir():
            files = sorted(p for p in Path(input_data).iterdir() if p.suffix.lower() in (".dat", ".fits") and "stacked" not in p.stem)
            stems = {p.stem for p in files if p.suffix.lower() == ".dat"}
            files = [p for p in files if p.suffix.lower() == ".dat" or p.stem not in stems]   # X.fits skipped if X.dat exists
            if not files:
                raise FileNotFoundError(f"No '.dat' or '.fits' spectra found in {input_data}.")
            input_data = files
        elif isinstance(input_data, (str, Path)):
            raise ValueError(f"{input_data} is not a directory. Give a directory or a list of spectrum files.")
        if not isinstance(input_data, (list, tuple)) or len(input_data) == 0:
            raise ValueError("input_data must be a directory, a list of spectrum files ('.dat' or '.fits') or a list of "
                             "(wavelength, flux) or (wavelength, flux, flux_err) tuples.")

        spectra = []
        for k, item in enumerate(input_data):
            if isinstance(item, (str, Path)):
                path = Path(item)
                if path.suffix.lower() not in (".dat", ".fits", ".fit", ".fts"):
                    raise ValueError(f"{path.name}: only '.dat' and '.fits' spectra can be stacked.")
                with contextlib.redirect_stdout(io.StringIO()):
                    wavelength, flux = self.load_calibrated_data(str(path), output_dir=output_dir, plot=False)
                spectra.append({"name": path.stem, "wavelength": wavelength.to(u.AA).value,
                                "flux": flux.to(FLAM_UNIT).value, "unit": FLAM_UNIT,
                                "flux_err": None if self.flux_err is None else self.flux_err.to(FLAM_UNIT).value,
                                "calibration_err": None if self.flux_calibration_error is None else self.flux_calibration_error.to(FLAM_UNIT).value,
                                "mask": self.spectrum_mask,
                                "wavelength_err": None if self.wavelength_systematic_error is None else self.wavelength_systematic_error.to(u.AA).value,
                                "instrumental_fwhm": None if getattr(self, "instrumental_fwhm", None) is None else self.instrumental_fwhm.to(u.AA).value})
            elif isinstance(item, (list, tuple)) and len(item) in (2, 3):
                wavelength = u.Quantity(item[0])
                wavelength = wavelength.value*u.AA if wavelength.unit == u.dimensionless_unscaled else wavelength.to(u.AA)
                flux = u.Quantity(item[1])
                unit = FLAM_UNIT if flux.unit == u.dimensionless_unscaled else flux.unit
                error = None
                if len(item) == 3 and item[2] is not None:
                    error = u.Quantity(item[2])
                    error = error.value if error.unit == u.dimensionless_unscaled else error.to(unit).value
                if len(wavelength) != len(flux) or (error is not None and len(error) != len(flux)):
                    raise ValueError(f"Spectrum {k}: wavelength, flux and flux_err must have the same length.")
                spectra.append({"name": f"spectrum_{k}", "wavelength": wavelength.value, "flux": flux.value, "unit": unit,
                                "flux_err": error, "calibration_err": None, "mask": None, "wavelength_err": None, "instrumental_fwhm": None})
            else:
                raise ValueError(f"Item {k} of input_data is neither a file path nor a (wavelength, flux[, flux_err]) tuple.")

        units = {spectrum["unit"] for spectrum in spectra}
        if len(units) > 1:
            try:
                for spectrum in spectra:
                    factor = (1*spectrum["unit"]).to(FLAM_UNIT).value
                    for key in ("flux", "flux_err", "calibration_err"):
                        if spectrum[key] is not None:
                            spectrum[key] = spectrum[key]*factor
                    spectrum["unit"] = FLAM_UNIT
            except u.UnitConversionError:
                raise ValueError(f"The spectra have incompatible flux units: {units}.")
        return spectra


    @_plot_style
    def stack_calib_spectra(self, input_data, method="median", target_name=None, output_dir="./", over_resolution_factor=1, save_file=False,
                            plot=True, plot_overlayed_spectra=True, normalize=None, normalization_window=None, reference="best",
                            redshifts=None, wavelength_range=None, min_spectra=1, clip_sigma=3.0, file_format="dat", return_errors=False):

        """
        This function stacks (co-adds) calibrated spectra, e.g. the different exposures of a target, or the spectra of several objects in
        their rest frames (a composite spectrum).

        The spectra are resampled onto a common linear wavelength grid conserving the flux, with their uncertainties and data quality
        masks; pixels flagged as bad (mask bits 1, 4, 8) or non-finite (e.g. inside saturated telluric bands) are left out. Optionally,
        the spectra are scaled to a common flux level (e.g. different slit losses) before being combined pixel by pixel.

        Parameters
        -----------
        input_data: list or string
            A directory with the spectra ('.dat' and '.fits' files; files with 'stacked' in the name are ignored), a list of spectrum
            files (written by extraction.target_flux_calibration(), extraction.telluric_correction() or previous easyspec versions), or a
            list of (wavelength, flux) or (wavelength, flux, flux_err) tuples (arrays or astropy Quantities).
        method: string
            "median" (default; robust against outliers, e.g. cosmic rays in one spectrum), "mean", "weighted" (inverse-variance weighted
            mean: the highest signal-to-noise, needs flux uncertainties) or "sigma_clip" (weighted mean, or mean without uncertainties,
            after rejecting, in each pixel, the points more than 'clip_sigma' away from the median; needs 3 or more spectra in the pixel).
        target_name: string
            Optional. Used in the plot title, in the output file name and in the subsequent analysis.
        output_dir: string
            The directory where the stacked spectrum file will be saved if save_file is True.
        over_resolution_factor: float
            The step of the common grid is the finest pixel size among the spectra divided by this factor. Default: 1. Values above 1
            do not add information: they only produce correlated pixels.
        save_file: bool
            If True, saves the stacked spectrum in the same format as the calibrated spectra (it can be read with load_calibrated_data()).
        plot: bool
            If True, plots the stacked spectrum.
        plot_overlayed_spectra: bool
            If True, the individual (resampled and, if requested, normalized) spectra are overlaid on the stacked one, and the number of
            spectra contributing to each pixel is shown below.
        normalize: None or string
            None (default): the fluxes are used as they are (the spectra of one target, all well calibrated). "reference": each spectrum is
            scaled to the reference spectrum (see 'reference'), keeping the flux calibration of the latter (corrects different slit losses
            or transparency). "unity": each spectrum is divided by its median flux, so the stacked spectrum is dimensionless (composites of
            different objects). The scale factors are ratios of medians inside 'normalization_window'.
        normalization_window: None or list
            The [lambda_min, lambda_max] interval (Angstrom, in the frame of the stack) used for the normalization, preferably with
            continuum only. Default: the region covered by all spectra (or, for "reference", the overlap of each spectrum with the reference).
        reference: "best", int or string
            The reference spectrum for normalize="reference": its index, its name, or "best" (the one with the highest median S/N).
        redshifts: None, float or list
            Optional. Redshift of each spectrum: the spectra are moved to the rest frame (lambda/(1+z), F_lambda*(1+z)) before stacking.
        wavelength_range: None or list
            Optional. [lambda_min, lambda_max] of the stacked spectrum. Default: from the bluest to the reddest pixel of all spectra.
        min_spectra: int
            Pixels with fewer than this number of contributing spectra are set to NaN (and flagged with bit 4 of the mask). Default: 1.
        clip_sigma: float
            Rejection threshold of method="sigma_clip", in units of the uncertainty of each point (or of the robust scatter). Default: 3.
        file_format: string
            "dat" (default), "fits" or "both".
        return_errors: bool
            If True, the 1-sigma uncertainty of the stacked flux is also returned.

        Returns
        --------
        wavelengths_stacked: numpy.ndarray (astropy.units Angstrom)
            The wavelengths of the common grid of the stacked spectrum (rest frame if 'redshifts' is given).
        stacked_flux: numpy.ndarray (astropy.units erg/cm2/s/A, or dimensionless with normalize="unity")
            The stacked flux density.
        stacked_flux_err: numpy.ndarray (only if return_errors=True)
            Its 1-sigma uncertainty: propagated from the flux errors of the spectra or, if these are not available, the standard error
            from the scatter between the spectra (NaN where only one spectrum contributes).

        Notes
        -----
        As in load_calibrated_data(), the uncertainty, calibration uncertainty, mask and wavelength accuracy of the stacked spectrum are
        stored in self.flux_err, self.flux_calibration_error, self.spectrum_mask and self.wavelength_systematic_error, so that the stacked
        spectrum can go directly to find_lines(), fit_lines(), etc. Details (scale factors, number of spectra per pixel, resampled
        spectra, chi2 of each spectrum around the mean) are stored in self.stack_info.
        """

        if method not in aux_stacking.STACK_METHODS:
            raise ValueError(f"Invalid stacking method '{method}'. Options are {aux_stacking.STACK_METHODS}.")
        if normalize not in (None, "reference", "unity"):
            raise ValueError("normalize must be None, 'reference' or 'unity'.")
        if file_format not in ("dat", "fits", "both"):
            raise ValueError("file_format must be 'dat', 'fits' or 'both'.")
        if int(min_spectra) < 1:
            raise ValueError("min_spectra must be at least 1.")

        spectra = self._stack_inputs(input_data, output_dir)
        n_spectra = len(spectra)
        names = [spectrum["name"] for spectrum in spectra]
        if n_spectra == 1:
            warnings.warn("Only one spectrum was given: it is just resampled onto the common grid.", stacklevel=3)

        # Rest frame:
        if redshifts is not None:
            z_values = np.atleast_1d(np.asarray(redshifts, dtype=float))
            if z_values.size == 1:
                z_values = np.repeat(z_values, n_spectra)
            if z_values.size != n_spectra or np.any(~np.isfinite(z_values)) or np.any(z_values <= -1):
                raise ValueError(f"'redshifts' must be one value or one per spectrum ({n_spectra}), all > -1.")
            for spectrum, z in zip(spectra, z_values):
                spectrum["wavelength"] = spectrum["wavelength"]/(1 + z)
                if spectrum["instrumental_fwhm"] is not None:
                    spectrum["instrumental_fwhm"] = spectrum["instrumental_fwhm"]/(1 + z)
                for key in ("flux", "flux_err", "calibration_err"):
                    if spectrum[key] is not None:
                        spectrum[key] = spectrum[key]*(1 + z)

        grid = aux_stacking.common_grid([s["wavelength"] for s in spectra], over_resolution_factor, wavelength_range)
        have_errors = all(spectrum["flux_err"] is not None for spectrum in spectra)
        have_calibration = all(spectrum["calibration_err"] is not None for spectrum in spectra)
        if method == "weighted" and not have_errors:
            raise ValueError("method='weighted' needs flux uncertainties for all spectra. Use 'median' or 'mean'.")
        if not have_errors and any(spectrum["flux_err"] is not None for spectrum in spectra):
            print("Note: some spectra have no flux uncertainties, so the uncertainty of the stack comes from the scatter between the spectra.")

        n_grid = len(grid)
        flux_matrix, error_matrix, calibration_matrix = np.full((n_spectra, n_grid), np.nan), np.full((n_spectra, n_grid), np.nan), np.full((n_spectra, n_grid), np.nan)
        mask_matrix = np.zeros((n_spectra, n_grid), dtype=np.int64)
        for i, spectrum in enumerate(spectra):
            new_flux, new_error, new_calibration, new_mask = aux_stacking.resample_spectrum(
                spectrum["wavelength"], spectrum["flux"], grid, spectrum["flux_err"] if have_errors else None,
                spectrum["calibration_err"], spectrum["mask"])
            flux_matrix[i], mask_matrix[i] = new_flux, new_mask
            if new_error is not None:
                error_matrix[i] = new_error
            if new_calibration is not None:
                calibration_matrix[i] = new_calibration
            if not np.any(np.isfinite(new_flux)):
                warnings.warn(f"{spectrum['name']} has no good pixels inside the stacked wavelength range.", stacklevel=3)

        # Normalization:
        factors = np.ones(n_spectra)
        reference_index = None
        if normalize is not None:
            if normalization_window is None:
                window = np.all(np.isfinite(flux_matrix), axis=0) if normalize == "unity" else np.ones(n_grid, dtype=bool)
                if normalize == "unity" and np.count_nonzero(window) < 5:
                    window = np.ones(n_grid, dtype=bool)
                    print("Note: the spectra do not share a common wavelength range, so each one is normalized by its median over all its pixels.")
            else:
                low, high = sorted(float(getattr(v, "value", v)) for v in normalization_window)
                window = (grid >= low) & (grid <= high)
            if normalize == "reference":
                if isinstance(reference, str) and reference == "best":
                    if have_errors:
                        with np.errstate(invalid="ignore", divide="ignore"), aux_stacking._quiet():
                            snr = np.nanmedian(flux_matrix/error_matrix, axis=1)
                    else:
                        snr = np.count_nonzero(np.isfinite(flux_matrix), axis=1).astype(float)
                    reference_index = int(np.nanargmax(snr))
                elif isinstance(reference, str):
                    if reference not in names:
                        raise ValueError(f"Reference '{reference}' not found. The spectra are {names}.")
                    reference_index = names.index(reference)
                else:
                    reference_index = int(reference)
                    if not 0 <= reference_index < n_spectra:
                        raise ValueError(f"'reference' must be between 0 and {n_spectra - 1}.")
            factors = aux_stacking.scale_factors(flux_matrix, error_matrix, normalize, reference_index, window)
            flux_matrix *= factors[:, None]
            error_matrix *= factors[:, None]
            calibration_matrix *= factors[:, None]

        stacked_flux, stacked_error, scatter_error, n_used, used = aux_stacking.combine(
            flux_matrix, error_matrix if have_errors else None, method=method, clip_sigma=clip_sigma)
        too_few = n_used < int(min_spectra)
        stacked_flux[too_few] = np.nan
        if stacked_error is None:
            stacked_error = scatter_error
        stacked_error = np.where(too_few, np.nan, stacked_error)

        # Calibration uncertainty (independent between spectra), with the weights of the combination:
        stacked_calibration = None
        if have_calibration:
            if method in ("weighted", "sigma_clip") and have_errors:
                weights = np.where(used, 1.0/error_matrix**2, 0.0)
            else:
                weights = used.astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                weights = weights/weights.sum(axis=0)
                stacked_calibration = np.sqrt(np.nansum((weights*np.where(used, calibration_matrix, 0.0))**2, axis=0))
            stacked_calibration[(n_used == 0) | too_few] = np.nan
            if method == "median":
                stacked_calibration *= np.where(n_used > 2, np.sqrt(np.pi/2), 1.0)

        stacked_mask = np.bitwise_or.reduce(np.where(used, mask_matrix, 0), axis=0) if n_spectra > 1 else mask_matrix[0].copy()
        stacked_mask[~np.isfinite(stacked_flux)] |= aux_stacking.NO_DATA_BIT
        # Instrumental resolution of the stack: mean of the resolutions of the spectra at each wavelength (if known for all of them)
        stacked_resolution = None
        if all(s["instrumental_fwhm"] is not None for s in spectra):
            resolutions = np.array([np.interp(grid, s["wavelength"], s["instrumental_fwhm"], left=np.nan, right=np.nan) for s in spectra])
            with np.errstate(invalid="ignore"), aux_stacking._quiet():
                stacked_resolution = np.nanmean(np.where(used, resolutions, np.nan), axis=0)
        wavelength_errors = [s["wavelength_err"] for s in spectra if s["wavelength_err"] is not None]
        wavelength_error = float(np.sqrt(np.mean(np.square(wavelength_errors)))) if wavelength_errors else None

        chi2 = aux_stacking.consistency_chi2(flux_matrix, error_matrix, stacked_flux, used) if (have_errors and n_spectra > 1) else np.full(n_spectra, np.nan)

        # Summary:
        unit = u.dimensionless_unscaled if normalize == "unity" else spectra[0]["unit"]
        coverage = np.count_nonzero(np.isfinite(stacked_flux))
        print(f"Stacked {n_spectra} spectra ({method}) onto {n_grid} pixels of {grid[1] - grid[0]:.3f} A "
              f"({grid[0]:.1f}-{grid[-1]:.1f} A{', rest frame' if redshifts is not None else ''}); {coverage} pixels with data.")
        for i, name in enumerate(names):
            details = []
            if normalize is not None:
                details.append("reference" if i == reference_index else f"scale factor {factors[i]:.4g}")
            if np.isfinite(chi2[i]):
                details.append(f"chi2_red around the mean = {chi2[i]:.2f}")
            if method == "sigma_clip":
                details.append(f"{np.count_nonzero(np.isfinite(flux_matrix[i]) & ~used[i])} pixels clipped")
            print(f"  {name}" + (": " + ", ".join(details) if details else ""))
        if np.any(chi2 > 3):
            warnings.warn("Some spectra disagree with the others well beyond their uncertainties (chi2_red > 3): source variability, "
                          "different slit losses/calibration (try normalize='reference'), wavelength offsets or underestimated errors.", stacklevel=3)

        # The stacked spectrum becomes the current spectrum of the analysis:
        self.output_dir = str(Path(output_dir))
        self.target_name = target_name if target_name is not None else ("stacked_spectrum" if n_spectra > 1 else names[0])
        self.flux_err = stacked_error*unit
        self.flux_systematic_error = self.flux_err
        self.flux_calibration_error = None if stacked_calibration is None else stacked_calibration*unit
        self.spectrum_mask = stacked_mask
        self.wavelength_systematic_error = None if wavelength_error is None else wavelength_error*u.AA
        self.instrumental_fwhm = None if stacked_resolution is None else stacked_resolution*u.AA
        self.stack_info = {"names": names, "method": method, "normalize": normalize, "scale_factors": factors,
                           "reference": None if reference_index is None else names[reference_index], "redshifts": redshifts,
                           "n_spectra_per_pixel": n_used, "resampled_fluxes": flux_matrix,
                           "resampled_errors": error_matrix if have_errors else None, "used": used,
                           "scatter_error": scatter_error, "chi2_red": chi2}

        if plot:
            title = f"Stacked spectrum of {target_name}" if target_name is not None else "Stacked spectrum"
            x_label = (r"Rest-frame $\lambda$ [$\AA$]" if redshifts is not None else r"Observed $\lambda$ [$\AA$]")
            y_label = r"F$_{\lambda}$ (normalized)" if normalize == "unity" else r"F$_{\lambda}$ [" + unit.to_string("latex_inline") + "]"
            if plot_overlayed_spectra and n_spectra > 1:
                fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05})
                for i in range(n_spectra):
                    ax.plot(grid, flux_matrix[i], linewidth=0.6, alpha=0.6, label=names[i] if n_spectra <= 10 else None)
            else:
                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax_n = None
            finite = np.isfinite(stacked_error)
            if np.any(finite):
                ax.fill_between(grid, stacked_flux - stacked_error, stacked_flux + stacked_error, color="gray", alpha=0.4,
                                step="mid", label=r"Stack $\pm1\sigma$")
            ax.plot(grid, stacked_flux, color="black", linewidth=0.8, label=f"Stacked spectrum ({method})")
            good = np.isfinite(stacked_flux)
            if np.any(good):
                low, high = np.nanpercentile(stacked_flux[good], [0.5, 99.8])
                ax.set_ylim(min(0, 1.2*low), 1.3*high)
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.minorticks_on()
            ax.grid(which="both", linestyle=":", linewidth=0.5)
            ax.legend(fontsize=9, ncol=2)
            if ax_n is not None:
                ax_n.step(grid, n_used, where="mid", color="C0", linewidth=0.8)
                ax_n.set_ylabel("N spectra")
                ax_n.set_ylim(0, n_spectra + 0.5)
                ax_n.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax_n.grid(which="both", linestyle=":", linewidth=0.5)
                ax_n.set_xlabel(x_label)
            else:
                ax.set_xlabel(x_label)
            plt.show()

        if save_file:
            base = Path(output_dir) / (f"{target_name}_stacked_{method}" if target_name is not None else f"stacked_spectrum_{method}")
            base.parent.mkdir(parents=True, exist_ok=True)
            calibration_column = stacked_calibration if stacked_calibration is not None else np.full(n_grid, np.nan)
            wavelength_column = np.full(n_grid, np.nan if wavelength_error is None else wavelength_error)
            flux_unit = "dimensionless" if normalize == "unity" else "erg/cm2/s/Angstrom"
            resolution_column = stacked_resolution if stacked_resolution is not None else np.full(n_grid, np.nan)
            if file_format in ("dat", "both"):
                np.savetxt(f"{base}.dat", np.c_[grid, stacked_flux, wavelength_column, stacked_error, calibration_column, stacked_mask,
                                                resolution_column],
                           fmt=["%.4f", "%.6e", "%.4f", "%.6e", "%.6e", "%d", "%.3f"],
                           header=f"wavelength (Angstrom), Flux ({flux_unit}), Systematic wavelength error (Angstrom), Flux error ({flux_unit}), "
                                  f"Calibration flux error ({flux_unit}), Mask (), Instrumental FWHM (Angstrom)")
            if file_format in ("fits", "both"):
                header = {"OBJECT": self.target_name, "NCOMBINE": n_spectra, "STACKMET": method,
                          "STACKNRM": str(normalize), "RESTFRAM": redshifts is not None}
                data_io.write_spectrum(f"{base}.fits", grid*u.AA, stacked_flux*unit, flux_err=stacked_error*unit, mask=stacked_mask,
                                       header=header, extra_columns={"flux_cal_err": calibration_column*unit,
                                                                     "wavelength_sys_err": wavelength_column*u.AA,
                                                                     "n_spectra": n_used, "instrumental_fwhm": resolution_column*u.AA})
            print(f"Stacked spectrum saved in {base}.{'dat/.fits' if file_format == 'both' else file_format}")

        if return_errors:
            return grid*u.AA, stacked_flux*unit, stacked_error*unit
        return grid*u.AA, stacked_flux*unit
