import numpy as np
import scipy
import matplotlib.pyplot as plt
import emcee
import corner
import glob
from pathlib import Path
import time
import warnings
from matplotlib.ticker import AutoMinorLocator
from scipy import interpolate
from easyspec.extraction import extraction
from astropy import units as u
from scipy.signal import medfilt
import platform
import os
from scipy.integrate import quad
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
from .line_models import *
from .aux_fit_lines import *
import re
import types

OS_name = platform.system()
plt.rcParams.update({'font.size': 12})
libpath = Path(__file__).parent.resolve() / Path("lines")
extraction = extraction()


class CombinedModel:
    """A picklable class that acts like a function representing one line model or combinations of line models"""
    
    def __init__(self, model_names):
        self.model_names = model_names
        self.param_counts = {
            'Gaussian': 3,  # mean, amplitude, std
            'Lorentz': 3,   # mean, amplitude, fwhm  
            'Voigt': 4,     # x_0, amplitude, fwhm_G, fwhm_L
            'Skewedgaussian': 4,  # peak location, amplitude, std, skewness
            'Skewedlorentzian': 4  # peak location, amplitude, fwhm, skewness
        }
    
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

    
    def continuum_fit(self,flux, wavelengths, continuum_regions, method = "powerlaw", pl_order=2, smooth_window=111):
    
        """
        This function fits the continuum emission with an spline or a power law.

        Parameters
        ----------
        flux: array (float)
            Array with the flux density to be fitted and (if spline is chosen) smoothed. 
        wavelengths: array (float)
            Wavelength must be in absolute values, i.e. without units.
        continuum_regions: list (float)
            The continuum will be computed only within these wavelength regions and then extrapolated everywhere else. E.g.: continuum_regions = [[3000,6830],[6930,7500]].
            If None, then all the wavelength range will be used.
        method: string
            This is the desired method to compute the continuum. Options are 'powerlaw' (or 'pl') and "median_filter".
            The 'powerlaw' method is better in case you have large emission/absorption lines. The method "median_filter" is excellent
            for extracting the continuum of a spectrum with narrow emission/absorption lines.
        pl_order: int
            Polynomial order for the power law fit. Used only if input variable method="powerlaw" (or "pl").
        smooth_window: int
            This is the size of the smooth window for the "median_filter" method. This number must be odd.

        Returns
        -------
        continuum_selection: array (float)
            Array with the continuum flux density.
        continuum_std_deviation: float
            The standard deviation for the continuum.
        """

        if continuum_regions is None:
            continuum_regions =[wavelengths[0],wavelengths[-1]]

        if isinstance(continuum_regions[0],float) or isinstance(continuum_regions[0],int):
            continuum_regions = [continuum_regions]

        if smooth_window%2 == 0:
            smooth_window = smooth_window - 1
            print(f"The input parameter 'smooth_window' must be odd. We are resetting it to {smooth_window}.")
        
        selection = np.asarray([])
        index_std_deviation = []
        for wavelength_region in continuum_regions:
            # Here we select the indexes of the wavelengths inside the intervals given by continuum_regions:
            index = np.where((wavelengths > wavelength_region[0]) & (wavelengths < wavelength_region[1]))[0]
            selection = np.concatenate([selection,index])
            index_std_deviation.append(index)
        
        selection = selection.astype(int)
        flux_continuum = flux[selection]
        wavelengths_continuum = wavelengths[selection]
        
        if method == "median_filter":
            tck = interpolate.splrep(wavelengths_continuum, flux_continuum,k=1)
            continuum_selection = interpolate.splev(wavelengths, tck)
            continuum_selection = medfilt(continuum_selection, smooth_window)

        elif method == "PL" or method == "powerlaw" or method == "pl" or method == "power-law":
            z = np.polyfit(wavelengths_continuum, flux_continuum, pl_order)
            continuum_selection = np.poly1d(z)
            continuum_selection = continuum_selection(wavelengths)
        else:
            raise RuntimeError("The input options for the 'method' variable are 'powerlaw' or 'median_filter'.")
        
        continuum_std_deviation = []
        for sub_region_indexes in index_std_deviation:
            continuum_std_deviation.append(np.std(flux[sub_region_indexes]-continuum_selection[sub_region_indexes]))

        return continuum_selection, continuum_std_deviation
        
    def load_calibrated_data(self, calibrated_spec_data, target_name = None, output_dir = "./",  plot = True):

        """
        This function loads the spectral calibrated data. Preferentially, you should use the file 'TARGETNAME_spec_X.dat' generated
        with easyfermi function 'extraction.target_flux_calibration()'.

        Parameters
        ----------
        calibrated_spec_data: string 
            The path to the data '.dat' file containing the calibrated spectrum.
        target_name: string
            Optional. This name will be used in all subsequent plots.
        output_dir: string
            A string with the path to the output directory. 
        plot: boolean
            If True, the spectrum will be shown.  

        Returns
        -------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        """

        self.output_dir = str(Path(output_dir))

        if target_name is None:
            target_name = calibrated_spec_data.split(".")[-1]
        self.target_name = target_name

        data = np.loadtxt(calibrated_spec_data)
        try:
            wavelengths, flux_density, wavelength_systematic_error, flux_density_sys_error = data[:,0]*u.angstrom, data[:,1]*u.erg / u.cm**2 / u.s / u.AA, data[:,2][0]*u.angstrom, data[:,3]*u.erg / u.cm**2 / u.s / u.AA  # Angstrom, erg/cm2/s/Angstrom, Angstrom, erg/cm2/s/Angstrom
            self.wavelength_systematic_error = wavelength_systematic_error
            self.flux_systematic_error = flux_density_sys_error
        except:
            wavelengths, flux_density = data[:,0]*u.angstrom, data[:,1]*u.erg / u.cm**2 / u.s / u.AA  # Angstrom, erg/cm2/s/Angstrom
            self.wavelength_systematic_error = None
            self.flux_systematic_error = None
        if plot:
            plt.figure(figsize=(12,5))
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(0,flux_density.value.max()*1.2)
            plt.title(f"Calibrated data -  {target_name}")
            plt.plot(wavelengths, flux_density, color='orange', label=target_name+" - calibrated data")
            plt.legend()
            plt.ylabel("F$_{\lambda}$ "+f"[{flux_density.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.show()

        
        self.wavelengths = wavelengths

        return wavelengths, flux_density

    def find_lines(self, wavelengths, flux_density, line_significance_sigma = 5, peak_distance = 30, peak_width = 10, method = "median_filter", continuum_regions = None,
                   pl_order=2, smooth_window=111, plot_lines = True, plot_regions = True, save_plot = False):

        """
        This function will find all emission/absorption lines with significance above 'line_significance_sigma' with respect to the local continuum.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        line_significance_sigma: float
            Defines how many standard deviations above the continuum the line peak must be in order to be detected.
        peak_distance: float
            The minimal distance (data bins, not in Angstroms) between peaks (>=1).
        peak_width: float
            Minimum required width of peaks in data bins. The number of data bins is equal to the number of pixels in the reduced spectral image.
        method: string
            This is the desired method to compute the continuum. Options are 'powerlaw' (or 'pl') and "median_filter".
            The 'powerlaw' method is better in case you have large emission/absorption lines. The method "median_filter" is excellent
            for extracting the continuum of a spectrum with narrow emission/absorption lines.
        continuum_regions: list (float)
            The continuum, in Angstroms, will be computed only within these wavelength regions and then extrapolated everywhere else. E.g.: continuum_regions = [[3000,6830],[6930,7500]].
            If None, then all the wavelength range will be used.
        pl_order: int
            Polynomial order for the power law fit. Used only if input variable method="powerlaw" (or "pl").
        smooth_window: int
            This is the size of the smooth window for the "median_filter" method. This number must be odd.
        plot_lines: boolean
            If True, the spectrum and all detected lines will be shown.
        plot_regions: boolean
            If True, the spectrum will be plotted in multiple regions (assuming continuum_regions is not None). For each one of these regions,
            the noise is independently estimated from the local continuum.
        save_plot: boolean
            If True, the spectrum plot will be saved in the output directory defined in analysis.load_calibrated_data().

        Returns
        -------
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A.
        line_std_deviation: numpy.ndarray (float)
            The standard deviation for the local continuum. Line significance is calculated with respect to this value.
        wavelength_peak_positions: numpy.ndarray (astropy.units Angstrom)
            The position of each peak in Angstroms.
        peak_heights: numpy.ndarray (astropy.units erg/cm2/s/A)
            The height of each peak in erg/cm2/s/A
        line_significance: array (floats)
            The line significance with respect to the local continuum standard deviation.
        """

        continuum_baseline, continuum_std_deviation = self.continuum_fit(flux_density.value, wavelengths.value, method = method, continuum_regions = continuum_regions, pl_order = pl_order, smooth_window = smooth_window) 

        peak_heights = np.asarray([])
        peak_position_index = np.asarray([])
        line_significance = np.asarray([])
        line_std_deviation = np.asarray([])
        line_position = []
        if plot_regions:
            plt.figure(figsize=(12,4))
            plt.ylabel("F$_{\lambda}$ "+f"[{flux_density.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            if continuum_regions is None:
                plt.title("Using the full wavelength range to estimate the continuum")
            else:
                plt.title("The continuum noise is independently estimated in each one of the vertical strips")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(-1.2*np.abs(np.min(flux_density.value-continuum_baseline)),1.2*np.max(flux_density.value-continuum_baseline))
        
        if continuum_regions is None:
            continuum_regions = [[np.min(wavelengths.value),np.max(wavelengths.value)]]

        # Below we do a loop over the values of standard deviation for the selected continuum regions. The line significance is estimated based on the standard deviation of the closest continuum region.
        ylim_min = 0
        for number,std_deviation in enumerate(continuum_std_deviation):
            peak_height = line_significance_sigma*std_deviation
            if number == 0:
                if len(continuum_std_deviation) > 1:
                    index = np.where(wavelengths.value < (continuum_regions[0][1] + continuum_regions[1][0])/2)[0]
                else:
                    index = np.asarray(range(len(wavelengths.value)))
            elif number != (len(continuum_std_deviation)-1):
                index = np.where((wavelengths.value > (continuum_regions[number-1][1] + continuum_regions[number][0])/2 ) & (wavelengths.value <  (continuum_regions[number][1] + continuum_regions[number+1][0])/2 ))[0]
            else:
                index = np.where(wavelengths.value > (continuum_regions[number-1][1] + continuum_regions[number][0])/2)[0]
            continuum_removed_flux = flux_density.value[index]-continuum_baseline[index]
            if plot_regions:
                plt.plot(wavelengths.value[index], continuum_removed_flux)
                plt.fill_betweenx(np.linspace(-1.2*np.abs(np.min(flux_density.value-continuum_baseline)),1.2*np.max(flux_density.value-continuum_baseline),10), continuum_regions[number][0],continuum_regions[number][1],alpha=0.3)#,color="gray")
            # Emission lines:
            local_peak_position_index, local_peak_heights = scipy.signal.find_peaks(continuum_removed_flux,height=peak_height,distance = peak_distance, width = peak_width)
            peak_heights = np.concatenate([peak_heights,local_peak_heights["peak_heights"]])
            peak_position_index = np.concatenate([peak_position_index, local_peak_position_index + index.min()])
            line_significance = np.concatenate([line_significance,local_peak_heights["peak_heights"]/std_deviation])
            line_std_deviation = np.concatenate([line_std_deviation, std_deviation*np.ones(len(local_peak_position_index))])
            if len(local_peak_position_index) > 0:
                emission_line_position = ["up"]*len(local_peak_position_index)
                line_position = line_position + emission_line_position
            # Absorption lines:
            local_peak_position_index, local_peak_heights = scipy.signal.find_peaks(-1*continuum_removed_flux,height=peak_height,distance = peak_distance, width = peak_width)
            peak_heights = np.concatenate([peak_heights,-1*local_peak_heights["peak_heights"]])
            peak_position_index = np.concatenate([peak_position_index, local_peak_position_index + index.min()])
            line_significance = np.concatenate([line_significance,local_peak_heights["peak_heights"]/std_deviation])
            line_std_deviation = np.concatenate([line_std_deviation, std_deviation*np.ones(len(local_peak_position_index))])
            if len(local_peak_position_index) > 0:
                local_ylim_min = -1.1*local_peak_heights["peak_heights"].max()
                if local_ylim_min < ylim_min:
                    ylim_min = local_ylim_min
                absorption_line_position = ["down"]*len(local_peak_position_index)
                line_position = line_position + absorption_line_position
            
        if len(peak_position_index) == 0:
            raise RuntimeError("No significant emission or absorption line was found. Maybe you can try playing with the input parameters 'line_significance_sigma' and 'peak_width'.")
        peak_position_index = peak_position_index.astype(int)
        peak_heights = (peak_heights+continuum_baseline[peak_position_index])*flux_density.unit
        wavelength_peak_positions = wavelengths[peak_position_index]
        continuum_removed_flux = flux_density.value-continuum_baseline

        if plot_lines:
            plt.figure(figsize=(12,5))
            if len(peak_heights) > 0:
                for number, peak_wavelength in enumerate(wavelength_peak_positions.value):
                    if line_position[number] == "up":
                        plt.text(peak_wavelength, peak_heights.value[number] + 0.05*peak_heights.value.max(), str(round(peak_wavelength,3))+"$\AA$", color="C0",rotation=90,fontsize=10, horizontalalignment="center", verticalalignment="bottom")
                    else:
                        text_height = np.mean(np.abs(peak_heights.value))
                        plt.text(peak_wavelength, continuum_baseline[peak_position_index][number] + text_height,str(round(peak_wavelength,3))+"$\AA$", color="red",rotation=90,fontsize=10, horizontalalignment="center", verticalalignment="bottom")
                        plt.vlines(peak_wavelength, peak_heights.value[number], continuum_baseline[peak_position_index][number] + 0.95*text_height,color="red")
            
            if method != "median_filter":
                plt.plot(wavelengths,continuum_baseline,label="Power-law continuum")
            else:
                plt.plot(wavelengths,continuum_baseline,label="Median-filter continuum")

            plt.plot(wavelengths, flux_density, color='orange')
            plt.plot(wavelengths, continuum_removed_flux, alpha=0.15, color='black', label="Continuum-subtracted spec")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(ylim_min,flux_density.value.max()*1.5)
            plt.ylabel("F$_{\lambda}$ "+f"[{flux_density.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.title(self.target_name)
            plt.legend()
            plt.tight_layout()
            if save_plot:
                plt.savefig(self.output_dir+f"/{self.target_name}_line_wavelengths.pdf",bbox_inches='tight')
        
        if plot_lines or plot_regions:
            plt.show()
        
        ordered_indexes = np.argsort(wavelength_peak_positions)
        peak_heights = peak_heights[ordered_indexes]
        wavelength_peak_positions = wavelength_peak_positions[ordered_indexes]
        line_significance = line_significance[ordered_indexes]

        return continuum_baseline, line_std_deviation, wavelength_peak_positions, peak_heights, line_significance
    

    def all_models(self, model_name):
        """
        Returns a picklable CombinedModel instance
        """
        model_name_list = re.findall('[A-Z][a-z_]*', model_name)
        return CombinedModel(model_name_list)

    def lnlike(self, theta, x, y, yerr, model):
        
        """
        Here we compute the likelihood of the current model given the data.
        """
        
        return -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)

    def lnprior(self, theta, priors):

        """
        This function checks if the input parameters satisfy the prior conditions.
        """
        if np.all((priors[:,0] < theta) & (theta < priors[:,1])):
            return 0.0
        else:
            return -np.inf
    

        
    def lnprob(self, theta, priors, x, y, yerr, model):
        lp = self.lnprior(theta, priors)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr, model)

    def line_MCMC(self, p0, priors, nwalkers, niter, initial, lnprob, data, model_name, custom_function=None, burn_in=100, show_progress=True):

        """This function runs a MCMC approach for one line (singular or blended) for a given a model."""
        
        priors = tuple([priors])  # The priors are transformed into a tuple containing only one element
        
        if model_name == "custom":
            if not callable(custom_function):
                raise Exception("Parameter custom_function is not callable. This parameter must be a function to work properly.")
                
        adopted_model = self.all_models(model_name)  # Here we choose a function for the fit
        adopted_model = tuple([adopted_model])  # The adopted model is transformed into a tuple containing only one element
        metadata = priors + data + adopted_model
        
        sampler = emcee.EnsembleSampler(nwalkers, len(initial), lnprob, args=metadata)
        
        start = time.time()
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=show_progress)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=show_progress)
        end = time.time()
        serial_time = end - start
        print("Single-core processing took {0:.1f} seconds".format(serial_time))

        return sampler, pos, prob, state
    

    def plotter(self, sampler, model_name, x, color="grey", normalization = 1):

        """In this function we plot the 'hairs' (i.e. alternative models) around the maximum likelihood model estimated with the MCMC method."""
                
        samples = sampler.flatchain
        adopted_model = self.all_models(model_name)
        
        x_plot = np.linspace(x.min(),x.max(),1000)
        for theta in samples[np.random.randint(len(samples), size=100)]:
            plt.plot(x_plot, adopted_model(theta, x_plot)*normalization, color=color, zorder=0, alpha=0.1)  # plotting with parameters in the posterior destribution

        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        plt.grid(linestyle=":")
        plt.legend()

    def MCMC_spread(self, x, samples, model_name, nsamples=100, custom_function=None):

        """
        Function to compute the median MCMC model and its standard deviation.
        """
        
        if model_name == "custom":
            if not callable(custom_function):
                raise Exception("Parameter custom_function is not callable. This parameter must be a function to work properly.")
        else:
            if custom_function is not None:
                custom_function = None
        
        models = []
        draw = np.floor(np.random.uniform(0,len(samples),size=nsamples)).astype(int)
        thetas = samples[draw]  # Each element of thetas contain the N parameters of the assumed model
        adopted_model = self.all_models(model_name)
        for theta in thetas:
            mod = adopted_model(theta,x)
            models.append(mod)
        spread = np.std(models,axis=0)
        median_model = np.median(models,axis=0)
        return median_model,spread
        
    def data_window_selection(self, wavelengths, spec_flux, spec_flux_err, line_region_min,line_region_max):
        
        """
        Function to slice the data into a wavelength window.

        Parameters
        ----------
        wavelengths: numpy.array
            This variable must be in absolute values, i.e. without units.
        spec_flux: numpy.array
            The spectral density array.
        spec_flux_err: numpy.array
            The spectral density array.
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
        x=wavelengths[selection]
        y=spec_flux[selection]
        yerr=spec_flux_err[selection] 
        return x, y, yerr

    def redshift_calculator(self, q_16,q_50,q_84,air_wavelength_line):
        z = (q_50 - air_wavelength_line) / air_wavelength_line
        zerror_down = (q_50-q_16)/air_wavelength_line
        zerror_up = (q_84-q_50)/air_wavelength_line
        return z, zerror_down, zerror_up

    def parameter_estimation(self, samples, air_wavelength_line=None, quantiles=[0.16, 0.5, 0.84], normalization = 1, parlabels=None, line_names="", output_dir=".", savefile=True):
        
        """
        In this function we estimate the parameter values and corresponding errors based on the 16%, 50%, and 84% quantiles of the
        MCMC posterior distributions of parameters.

        Parameters
        ----------
        samples: list
            A list containing the MCMC posterior distributions.
        air_wavelength_line: float
            The line rest-frame wavelength used to compute the redshift.
        quantiles: list
            The quantiles adpted to compute the parameter values and corresponding errors. Default is quantiles=[0.16, 0.5, 0.84].
        normalization: float
            The plot normalization.
        parlabels: list
            This is a list with the names of the free parameters in the adopted model. If you use a model with two peaks,
            then parlabels needs two input lists, e.g.: parlabels=[['a','b'],['c','d']]. If it has three peaks, parlabels
            must be parlabels=[['a','b'],['c','d'],['e','f']] and so on.
        line_names: string or list of strings
            The line names.
        outputdir: string
            The output directory.        
        savefile: boolean
            If True, the data will be saved in the output directory.
        
        Returns
        -------
        par_values: list
            A list with the parameter values recovered from the 50% quantiles of the MCMC posterior distributions of parameters.
        par_values_errors: list
            A list with the parameter errors recovered from the 16% and 84% quantiles of the MCMC posterior distributions of parameters.
        par_names: list
            A list with the parameter names.

        """
        
        if isinstance(parlabels[0],str):
            parlabels = [parlabels]
            
        output_dir = str(Path(output_dir))
        
         
        # If air_wavelength_line is a number, we transform it into a list with a single element: 
        if air_wavelength_line is not None: 
            if isinstance(air_wavelength_line,float) or isinstance(air_wavelength_line,int):
                air_wavelength_line = [air_wavelength_line]
            elif isinstance(air_wavelength_line,np.ndarray) or isinstance(air_wavelength_line,list):
                pass
            else:
                raise Exception("The input value for air_wavelength_line must be a float, an integer, a list, or a numpy array.")
        
        # Checking line_names:
        if isinstance(line_names,str):
                line_names = [line_names]
        elif isinstance(line_names,np.ndarray) or isinstance(line_names,list):
            pass
        else:
            raise Exception("The input value for line_names must be a string, a list, or a numpy array.")
        
        previous_j = 0
        par_values = []
        par_values_errors = []
        par_names = []
        for i in range(len(air_wavelength_line)):
            if savefile:
                f = open(f'{output_dir}/{self.target_name}_{line_names[i]}_line_fit_results.csv','w')
                f.write("# Parameter name, value, error_down, error_up\n")
            ndim = len(parlabels[i])  # number of dimensions/parameters
            parlabels[i] = list(parlabels[i])

            for j in range(ndim):  # must be done once per variable
                q_16, q_50, q_84 = corner.quantile(samples[:,j+previous_j], quantiles)
                dx_down, dx_up = q_50-q_16, q_84-q_50
                # Computing the redshift of the line:
                if parlabels[i][j] == "Location" and air_wavelength_line is not None:
                    z, zerror_down, zerror_up = self.redshift_calculator(q_16,q_50,q_84,air_wavelength_line[i])
                    if savefile:
                        f.write(f"z, {z}, {zerror_down}, {zerror_up}\n")
                    par_values.append(z)
                    par_values_errors.append([zerror_down, zerror_up])
                    par_names.append("redshift")

                if parlabels[i][j][0:9] == "Amplitude":  # If the variable is the amplitude, then we have to normalize the data
                    par_values.append(q_50*normalization)
                    par_values_errors.append([dx_down*normalization, dx_up*normalization])
                elif parlabels[i][j][0:3] == "std":  # For Gaussian fits, we convert the std to FWHM
                    parlabels[i][j] = "fwhm_Gauss"
                    fwhm = q_50*2*np.sqrt(2 * np.log(2))
                    fwhm_error_down = dx_down*2*np.sqrt(2 * np.log(2))
                    fwhm_error_up = dx_up*2*np.sqrt(2 * np.log(2))
                    par_values.append(fwhm)
                    par_values_errors.append([fwhm_error_down, fwhm_error_up])
                else:
                    par_values.append(q_50)
                    par_values_errors.append([dx_down, dx_up])
                par_names.append(parlabels[i][j])
                
                if savefile:
                    f.write(f"{parlabels[i][j]}, {par_values[-1]}, {par_values_errors[-1][0]}, {par_values_errors[-1][1]}\n")
            
            previous_j = j + previous_j + 1
            if savefile:
                f.close()
        
        return par_values, par_values_errors, par_names


    def quick_plot(self, x, y, model_name, parlabels, sampler=None, best_fit_model=None, theta_max=None, normalization = 1, hair_color="grey", title="",xlabel="Observed $\lambda$ [$\AA$]",ylabel="F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ ]",overplot_median_model=True,savefig=True, outputdir="./"):
        
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
            The MCMC sampler.
        best_fit_model: numpy.array
            Array with the best-fit model.
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
            If True, the median model and model standard deviation will be overploted.
        savefig: boolean
            If True, the figure will be saved in the output directory.
        outputdir: string
            The output directory.

        Returns
        -------

        """
                
        f = plt.figure(figsize=(10,8))
        ax = f.add_subplot(1,1,1)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='major', length=5, direction='in')
        ax.tick_params(which='minor', length=2.5, direction='in',bottom=True, top=True, left=True, right=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)

        plt.plot(x,y*normalization,color="orange", label="Data after cont. subtraction")
        if sampler is not None:
            self.plotter(sampler,model_name,x,color=hair_color, normalization = normalization)
        if best_fit_model is not None:
            x_plot = np.linspace(x.min(),x.max(),len(best_fit_model))
            plt.plot(x_plot,best_fit_model*normalization, color="black", label="Highest likelihood model")
        if overplot_median_model:
            samples = sampler.flatchain
            median_model,spread = self.MCMC_spread(x, samples, model_name, nsamples=200)
            x_plot = np.linspace(x.min(),x.max(),len(median_model))
            plt.plot(x_plot,median_model*normalization, color="C0", ls="--", label="Median model")
            plt.fill_between(x_plot,(median_model+spread)*normalization,(median_model-spread)*normalization,color="C0",alpha=0.3,label="$1\sigma$")

        plt.xlabel(xlabel,fontsize=12)
        plt.ylabel(ylabel,fontsize=12)
        plt.title(title)
        plt.grid(which="both", linestyle=":")
        plt.ticklabel_format(scilimits=(-5, 8))
        if normalization < 0:
            plt.ylim(1.1*y.max()*normalization,(y.min()-0.1*y.max())*normalization)
        else:
            plt.ylim((y.min()-0.1*y.max())*normalization,1.1*y.max()*normalization)
        
        if isinstance(parlabels,list):
            parlabels = np.asarray(parlabels)
        peak_indexes = np.where(parlabels=="Location")[0]

        for i in peak_indexes:
            plt.vlines(theta_max[i],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
        plt.legend()
        if savefig:
            plt.savefig(outputdir+"/"+title+"_line.png")
        return

    def insert_rows(self, array, indices, row):
        """Insert one or more copies of 'row' into 'array' at given indices (axis=0)."""
        array = np.asarray(array)
        row = np.asarray(row)

        # Ensure indices are sorted descending
        indices = np.sort(indices)[::-1]

        # Clamp indices that are out of range
        indices = np.clip(indices, 0, len(array))

        for idx in indices:
            array = np.insert(array, idx, row, axis=0)

        return array

    def merge_fit_results(self, target_name, list_of_files=None, output_dir="./"):
        
        """
        This function merges the individual data files for each line into a single merged data file.
        """
        
        output_dir = str(Path(output_dir))
        
        if list_of_files is None:
            list_of_files = glob.glob(output_dir+"/*_line_fit_results.csv")
        else:
            list_of_files = np.genfromtxt(list_of_files,dtype=str)
            
        f = open(f'{output_dir}/'+target_name+"_lines.csv","w")
        data_list = []
        line_names = []
        maximum_number_of_pars = 0
        index_maximum_number_of_pars = 0
        # Finding the line data file with the largest number of parameters: 
        for n,line_file in enumerate(list_of_files):
            if line_file[-16:] == "custom_model.csv":
                print(f"Skipping {line_file}.")
                continue
            
            line_data = np.genfromtxt(line_file,delimiter=",",dtype=str)
            data_list.append(line_data)
            line_names.append(line_file.split("/")[-1][:-21])
            if len(line_data) > maximum_number_of_pars:
                maximum_number_of_pars = len(line_data)
                index_maximum_number_of_pars = n
        
        

        # Write header:
        parameter_names = np.asarray(data_list[index_maximum_number_of_pars][:,0])
        minimum_list_of_parameter_names = np.asarray(['z','Location','Amplitude','fwhm_Lorentz','fwhm_Gauss','skewness'])
        if len(parameter_names) < len(minimum_list_of_parameter_names):
            parameter_names = minimum_list_of_parameter_names
        f.write("# Line name")
        for parameter_name in parameter_names:
            if parameter_name == "Location":
                f.write(", obs_"+parameter_name+" [Ang], obs_"+parameter_name+"_error_down, obs_"+parameter_name+"_error_up")
            elif parameter_name == "Amplitude":
                f.write(", obs_"+parameter_name+" (flux_dens - continuum) [erg cm-2 s-1 Ang-1], obs_"+parameter_name+"_error_down, obs_"+parameter_name+"_error_up")
            elif parameter_name[0:4] == "fwhm":
                f.write(", obs_"+parameter_name+" [Ang], obs_"+parameter_name+"_error_down, obs_"+parameter_name+"_error_up")
            elif parameter_name == "skewness":
                f.write(", obs_"+parameter_name+", obs_"+parameter_name+"_error_down, obs_"+parameter_name+"_error_up")
            else:
                f.write(", "+parameter_name+", "+parameter_name+"_error_down, "+parameter_name+"_error_up")
        
        if self.wavelength_systematic_error is not None:
            f.write(", Systematic wavelength error [Ang]")

        if self.flux_systematic_error is not None:
            f.write(", Systematic amplitude error (erg/cm2/s/Angstrom)")

        # Writing down the parameters and filling the empty spaces with zeros:
        for i in range(len(data_list)):

            f.write("\n"+line_names[i])
            line_array = np.asarray(data_list[i])

            mask_in_par_names = np.isin(parameter_names, line_array[:,0])
            # Invert mask to get elements missing in parameter_names
            mask_not_in_par_names = ~mask_in_par_names

            # Indices in parameter_names of elements NOT in line_array[:,0]
            idx_missing = np.where(mask_not_in_par_names)[0]
            

            # Insert element 3 at index 2
            new_row = np.array(["missing","0","0","0"])
            # Repeat the new row

            # Sort indices descending to avoid shifting
            line_array = self.insert_rows(line_array, idx_missing, new_row)

            for parameter in line_array:
                f.write(", "+parameter[1]+", "+str(np.abs(float(parameter[2])))+", "+str(np.abs(float(parameter[3]))))
                

            if self.wavelength_systematic_error is not None:
                f.write(f", {self.wavelength_systematic_error.value}")
            
            index = extraction.find_nearest(self.wavelengths.value,float(line_array[1][1]))

            if self.flux_systematic_error is not None:
                f.write(f", {self.flux_systematic_error.value[index]}")

        f.close()
        return

    def parameter_time_series(self, initial, sampler, labels):
        
        """
        This function plots the time series of the parameters running in the MCMC.
        """

        fig, axes = plt.subplots(len(initial), figsize=(10, 2*len(initial)), sharex=True)
        samples = sampler.get_chain()
        for i in range(len(initial)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        return
    
    def automatic_priors(self, which_model, observed_wavelength, peak_height, line_region_min, line_region_max):

        """
        In this function we automatically set the priors, initial parameter values and select the line model.

        Parameters
        ----------
        which_models: string
            The models to be applied to the line. Options are "Gaussian", "Lorentz" and "Voigt".
        observed_wavelength: float
            The wavelength corresponding to the peak position.
        peak_height: flaot
            The height of the peak with respect to the continuum emission.
        line_region_min: float
            The starting wavelength around the studied line. 
        line_region_max: float
            The final wavelength around the studied line. 
        
        Returns
        -------
        initial: numpy.array
            An array with the automatic initial parameter values for the MCMC.
        priors: numpy.array
            An array containing three or four lists of two elements, i.e. one list for each parameter given in 'initial'.
        labels: list
            A list with the names of each parameter.
        adopted_model: method
            A function with the adopted model, i.e., a "Gaussian", "Lorentz" or "Voigt".
        """
        
        if which_model == "Gaussian":
            initial = np.array([observed_wavelength, peak_height, 10])
            if peak_height > 0:  # This step is necessay because the priors must always go fro mthe smallest value up to the largest.
                priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150]])
            else:
                priors = np.array([[line_region_min,line_region_max],[10*peak_height,0.1*peak_height],[0.1,150]])
            labels = ["Location", "Amplitude", "std"]
            adopted_model = model_Gauss
        elif which_model == "Lorentz":
            initial = np.array([observed_wavelength, peak_height, 10])
            if peak_height > 0:
                priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150]])
            else:
                priors = np.array([[line_region_min,line_region_max],[10*peak_height,0.1*peak_height],[0.1,150]])
            labels = ["Location", "Amplitude", "fwhm_Lorentz"]
            adopted_model = model_Lorentz
        elif which_model == "Voigt":
            initial = np.array([observed_wavelength, peak_height, 10, 10])
            if peak_height > 0:
                priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150],[0.1,150]])
            else:
                priors = np.array([[line_region_min,line_region_max],[10*peak_height,0.1*peak_height],[0.1,150],[0.1,150]])
            labels = ["Location", "Amplitude", "fwhm_Lorentz", "fwhm_Gauss"]
            adopted_model = model_Voigt
        elif which_model == "Skewedgaussian":
            initial = np.array([observed_wavelength, peak_height, 10, 0])
            if peak_height > 0:
                priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150],[-50,50]])
            else:
                priors = np.array([[line_region_min,line_region_max],[10*peak_height,0.1*peak_height],[0.1,150],[-50,50]])
            labels = ["Location", "Amplitude", "std", "skewness"]
            adopted_model = model_skewed_gaussian
        elif which_model == "Skewedlorentzian":
            initial = np.array([observed_wavelength, peak_height, 10, 0])
            if peak_height > 0:
                priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150],[-50,50]])
            else:
                priors = np.array([[line_region_min,line_region_max],[10*peak_height,0.1*peak_height],[0.1,150],[-50,50]])
            labels = ["Location", "Amplitude", "fwhm_Lorentz", "skewness"]
            adopted_model = model_skewed_lorentzian
        
        return initial, priors, labels, adopted_model
    
    def estimate_norm_height(self, flux_density, continuum_baseline, wavelength_peak_position, peak_height):

        """
        Funtion to estimate the normalized height of a peak to set the priors for the MCMC.

        Parameters
        ----------
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        wavelength_peak_position: astropy.units Angstrom
            The position of the peak in the wavelength axis in Angstroms.
        peak_height: astropy.units erg/cm2/s/A
            The height of the peak in erg/cm2/s/A.

        Returns
        -------
        norm_height: float
            The estimated normalized height of the input peak. This value can be set to estimate priors for the MCMC.
        
        """

        if isinstance(wavelength_peak_position, u.quantity.Quantity):
            wavelength_peak_position = wavelength_peak_position.value

        local_normalization = 10**round(np.log10(np.median(flux_density.value - 0.9 * continuum_baseline)))

        norm_height = peak_height/local_normalization

        return norm_height
    

    def calculate_continuum_subtracted_heights(self,wavelengths, peak_positions, peak_heights, continuum_baseline):
        """Calculate peak heights relative to local continuum."""
        # Find indices closest to each peak position
        continuum_indices = [
            extraction.find_nearest(wavelengths.value, pos) 
            for pos in peak_positions
        ]
        
        # Subtract continuum baseline from peak heights
        continuum_values = continuum_baseline[continuum_indices]
        return peak_heights - continuum_values

    def fit_lines(self, wavelengths, flux_density, continuum_baseline, wavelength_peak_positions, rest_frame_line_wavelengths, peak_heights, line_std_deviation,
                  blended_line_min_separation = 50, which_models="Lorentz", line_names = None, overplot_archival_lines = ["H"], priors = None, MCMC_walkers = 100,
                  MCMC_iterations = 200, MCMC_burn_in=100, MCMC_show_progress=True, plot_spec = True, plot_MCMC = False, overplot_median_model = False, save_results = True):

        """
        Perform Markov Chain Monte Carlo (MCMC) fitting of spectral lines to estimate 
        line parameters and their uncertainties.
        
        This function fits Gaussian, Lorentzian, or Voigt profiles to spectral lines 
        using MCMC methods, with support for blended lines and parallel processing 
        for Voigt profile calculations.
        
        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        wavelength_peak_positions: numpy.ndarray (astropy.units Angstrom)
            The position of each peak in Angstroms found with the function analysis.find_lines().
        rest_frame_line_wavelengths: list
            A list with the rest frame wavelength values for each one of the input lines.
        peak_heights: numpy.ndarray (astropy.units erg/cm2/s/A)
            The height of each peak in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        line_std_deviation: numpy.ndarray (float)
            The standard deviation for the local continuum. This variable is an output of the function analysis.find_lines().
        blended_line_min_separation: float
            The minimum separation between blended line peaks in Angstrom. If the line peaks are closer than this value, a blended model will be adopted.
        which_models: string or list of strings
            A list containing the models to be applied to each line, e.g.: if you are trying to model 2 lines, you can use which_models = ["Lorentz","Gaussian"].
            If you wish to use the same model for all lines, you can use which_models = ["Gaussian","Gaussian"] or simply which_models = "Gaussian". Options are
            "Gaussian", "Lorentz" and "Voigt".
        line_names: list
            Optional. A list with line names, e.g.: if you are trying to model 2 lines, you can use line_names = ["Hbeta","Halpha"].
            If the names are not provided, easyspec will call the lines line_0, line_1... and so on.
        overplot_archival_lines: list
            List with the strings corresponding to different elements. E.g.: ["H","He"] will overplot all the Hydrogen and Helium lines redshifted based on the average
            redshift of the lines given in wavelength_peak_positions. If you don't want to overplot lines, use overplot_archival_lines = None. If you use too many lines
            as input, they will very likely overlap in the plot. Be aware that this feature is meant only for guiding the user! There are several lines which are not
            included in our database, but we tried to select the most commonly seen lines in galaxy, quasar and stellar spectra.
        priors: list or list of lists
            This parameter is complicated. It is better if you leave it as "None". This parameter controls the priors used in the MCMC in the estimation of the
            line parameters. The initial parameters for the MCMC are always defined as wavelength_peak_positions (for the position of the line peak) and 
            peak_heights (for the height of the peak). If priors = None, the priors are set to wavelength_peak_positions +- 100 Angstroms (or to half the distance
            to the closest line if this line is closer than 100 Angstroms), 0.1*peak_heights up to 10*peak_heights (normalized based on the continuum),
            and the std or fwhm are confined within 0.1 to 150. If you are e.g. analysing 4 lines with the Lorentz model and want to change the priors of the third
            line, you can set priors = [None, None,[[7496,7696],[0.1, 150],[2,150]], None], where the list of three ranges here represents the allowed sampling
            intervals, i.e. the position of the peak, the peak heigh in terms of the continuum level and the fwhm. For the Voigt model, the input variable would
            be  priors = [None, None,[[7496,7696],[0.1, 150],[2,150],[1,150]], None]. Of course you can set up the ranges for all lines.
        MCMC_walkers: int
            This is the number of walkers for the MCMC. Default = 100.
        MCMC_iterations: int
            This is the number of iterations for the MCMC. Default = 200.
        MCMC_burn_in: int
            The MCMC burn-in is the initial phase of the simulation where the early iterations are discarded. Default = 100.
        MCMC_show_progress: boolean
            If True, the progess bars for the MCMC are shown. Default = True.
        plot_spec: boolean
            If True, a plot of the spectrum with the lines requested in the input variable overplot_archival_lines will be shown.
        plot_MCMC: boolean
            If True, a series of diagnostic plots for the MCMC will be shown, as the corner plot, the evolution of the parameters over time, and the line fitted to 
            the data.
        overplot_median_model: boolean
            If True, the median model and model standard deviation will be overploted in the diagnostic plots.
        save_results: boolean
            If True, the plots and fit information will be saved in the output directory defined in the function analysis.load_calibrated_data()
            
        Returns
        -------
        par_values_list: list
            This is a list containing sublists with the best-fit values for each line model.
        par_values_errors_list: list
            A list with the asymmetrical errors for each parameter listed in par_values_list.
        par_names_list: list
            A list with the names of all the parameters used in each line model.
        samples_list: list
            A list containing the MCMC posterior distributions.
        line_windows: list
            A list containing the wavelength intervals around each line.
        XXXXXXX_lines.csv:
            Optional. This file contains all the best-fit parameters for each line model and is saved in the output directory defined in the
            function analysis.load_calibrated_data(). "XXXXXXX" here stands for the target's name.
        """

        # Validate and preprocess input parameters
        aux_validate_line_names(line_names)
                
        # Convert all inputs to consistent numpy arrays with proper shape handling
        wavelength_peak_positions = aux_ensure_numpy_array(wavelength_peak_positions, 'wavelength_peak_positions')
        peak_heights = aux_ensure_numpy_array(peak_heights, 'peak_heights')
        line_std_deviation = aux_ensure_numpy_array(line_std_deviation, 'line_std_deviation')
        rest_frame_line_wavelengths = aux_ensure_numpy_array(rest_frame_line_wavelengths, 'rest_frame_line_wavelengths')

        # Validate and expand model specifications
        which_models = aux_validate_models(which_models, n_lines = len(rest_frame_line_wavelengths))
        copy_which_models = which_models.copy()

        # Generate default line names if not provided
        line_names = aux_generate_line_names(line_names, len(rest_frame_line_wavelengths))
        self.line_names = line_names

        # Calculate continuum-subtracted peak heights
        peak_heights = self.calculate_continuum_subtracted_heights(wavelengths, wavelength_peak_positions, peak_heights, continuum_baseline)

        # Calculate normalization factor for the spectrum
        local_normalization = 10**round(np.log10(np.median(flux_density.value - 0.9 * continuum_baseline)))

        # Initialize default priors if none provided
        if priors is None:
            priors = [None] * len(wavelength_peak_positions)

        # Here we identify the blended lines:
        peak_distances = np.diff(wavelength_peak_positions)
        blended_line = [False]  # The first line will never be blended with a precedent line
        for distance in peak_distances:
            if distance < blended_line_min_separation:
                blended_line.append(True)
            else:
                blended_line.append(False)
        
        blended_line.append(False)  # We append a last element to the blended_line list to guarantee that our list can go up to number+1 and the last line can be fitted.

        par_values_list, par_values_errors_list, par_names_list, samples_list, line_windows = [], [], [], [], []
        line_region_min_cache, initial_cache, p0_cache, priors_cache, labels_cache = [], [], [], [], []
        for number, peak_height in enumerate(peak_heights):
            continuum_subtracted_flux = (flux_density.value - continuum_baseline)/local_normalization
            peak_height = peak_height/local_normalization
            local_line_std_deviation = line_std_deviation[number]/local_normalization

            if priors is None or priors[number] is None:
                 # Automatic priors: calculate line region from peak positions
                line_region_min, line_region_max = aux_calculate_line_region(wavelength_peak_positions, number, len(rest_frame_line_wavelengths))
                line_region_min_cache.append(line_region_min)
                
                initial, local_priors, labels, adopted_model = self.automatic_priors(copy_which_models[number], wavelength_peak_positions[number],
                                                                                     peak_height, line_region_min, line_region_max)

            else:
                # In the case of a single-line analysis, if the user inputs priors=[[7500,7700],[0.1],[2,50]] instead of priors=[ [[7500,7700],[0.1],[2,50]] ], the analysis will work anyway.
                if np.isscalar(priors[number][0]) or isinstance(priors[number][0], (int, float)):
                    priors = [priors]
                # User-provided priors - get basic values first
                initial, local_priors, line_region_min, line_region_max = aux_parse_custom_priors(priors[number], wavelength_peak_positions[number], peak_height)

                # Then call automatic_priors separately
                _, _, labels, adopted_model = self.automatic_priors(copy_which_models[number], wavelength_peak_positions[number], peak_height, 
                                                                    line_region_min=None, line_region_max=None)
            
            # Reseting wavelength windows for blended lines:
            line_region_min, line_region_max = aux_resetting_wavelength_windows(number,rest_frame_line_wavelengths,priors,blended_line,wavelength_peak_positions,line_region_min_cache, line_region_min, line_region_max)


            line_windows.append([line_region_min, line_region_max])
            x,y,yerr = self.data_window_selection(wavelengths.value, continuum_subtracted_flux, local_line_std_deviation*np.ones(len(wavelengths.value)), line_region_min, line_region_max)
            
            data = (x, y, yerr)
            p0 = [np.array(initial) + 0.01 * np.random.randn(len(initial)) for i in range(MCMC_walkers)]  # p0 is the methodology of stepping from one place on a grid to the next.
            initial_cache.append(initial)
            p0_cache.append(p0)
            priors_cache.append(local_priors)
            labels_cache.append(labels)
            
            if blended_line[number] is False:
                blended_line_names = []
                blended_rest_frame_line_wavelengths = []
                blended_labels = []
                if blended_line[number+1] is False:
                    sampler, _, _, _ = self.line_MCMC(p0, local_priors, MCMC_walkers, MCMC_iterations, initial, self.lnprob, data, copy_which_models[number], burn_in=MCMC_burn_in, show_progress=MCMC_show_progress)
                    samples = sampler.flatchain
                    samples_list.append(samples)
                    theta_max = samples[np.argmax(sampler.flatlnprobability)]

                    par_values, par_values_errors, par_names = self.parameter_estimation(samples, air_wavelength_line = rest_frame_line_wavelengths[number],
                                                                                        normalization=local_normalization, parlabels = list.copy(labels), line_names=line_names[number],
                                                                                        output_dir = self.output_dir, savefile=save_results)
                    par_values_list.append(par_values)
                    par_values_errors_list.append(par_values_errors)
                    par_names_list.append(par_names)

                    x_best_fit = np.linspace(x.min(),x.max(),1000)
                    best_fit_model = adopted_model(theta_max, x_best_fit)

                    if plot_MCMC:
                        corner.corner(
                            samples,
                            show_titles=True,
                            labels=labels,
                            plot_datapoints=True,
                            quantiles=[0.16, 0.5, 0.84],
                        )
                        plt.suptitle(line_names[number]+"\n(normalized amplitude)", x=0.7)


                        self.parameter_time_series(initial, sampler, labels)

                        self.quick_plot(x,y,model_name=copy_which_models[number], parlabels=labels, sampler=sampler,best_fit_model=best_fit_model,theta_max=theta_max, normalization=local_normalization,
                                        hair_color="grey",title=self.target_name+" - "+line_names[number], ylabel="F$_{\lambda} - F_{continuum}$ ["+f"{flux_density.unit}]", overplot_median_model=overplot_median_model, outputdir = self.output_dir)

                else:
                    blended_line_names.append(line_names[number])
                    blended_rest_frame_line_wavelengths.append(rest_frame_line_wavelengths[number])
                    blended_labels.append(labels_cache[number])
                    continue  # If the next line is blended, we skip this step of the loop

            elif blended_line[number] is True:
                copy_which_models[number] = copy_which_models[number-1] + copy_which_models[number]
                initial_cache[number] = np.concatenate([initial_cache[number-1], initial_cache[number]])
                p0_cache[number] = np.concatenate([p0_cache[number-1], p0_cache[number]],axis=1)
                priors_cache[number] = np.concatenate([priors_cache[number-1], priors_cache[number]])
                blended_labels.append(labels_cache[number]) # = np.concatenate([[labels_cache[number-1]], [labels_cache[number]]])
                blended_rest_frame_line_wavelengths.append(rest_frame_line_wavelengths[number])
                blended_line_names.append(line_names[number])


                if blended_line[number+1] is False:
                    p0 =  p0_cache[number]
                    initial = initial_cache[number]
                    local_priors = priors_cache[number]
                    labels = blended_labels #list(labels_cache[number])

                    

                    sampler, _, _, _ = self.line_MCMC(p0, local_priors, MCMC_walkers, MCMC_iterations, initial, self.lnprob, data, copy_which_models[number], burn_in=MCMC_burn_in, show_progress=MCMC_show_progress)
                    samples = sampler.flatchain
                    samples_list.append(samples)
                    theta_max = samples[np.argmax(sampler.flatlnprobability)]

                    par_values, par_values_errors, par_names = self.parameter_estimation(samples, air_wavelength_line = blended_rest_frame_line_wavelengths,
                                                                                        normalization=local_normalization, parlabels = list.copy(labels), line_names=blended_line_names,
                                                                                        output_dir = self.output_dir, savefile=save_results)

                    
                    par_names.append("redshift")  # This is just to facilitate the loop below.
                    redshift_index = np.where(np.asarray(par_names)=="redshift")[0]
                    for i in range(len(redshift_index)-1):
                        par_values_list.append(par_values[redshift_index[i]:redshift_index[i+1]])
                        par_values_errors_list.append(par_values_errors[redshift_index[i]:redshift_index[i+1]])
                        par_names_list.append(par_names[redshift_index[i]:redshift_index[i+1]])

                    x_best_fit = np.linspace(x.min(),x.max(),1000)
                    adopted_model = self.all_models(copy_which_models[number])
                    best_fit_model = adopted_model(theta_max, x_best_fit)


                    labels_corner = np.asarray([])
                    line_names_corner = ""
                    for i,j in zip(labels,blended_line_names):
                        labels_corner = np.concatenate([labels_corner,i])
                        if line_names_corner == "":
                            line_names_corner = line_names_corner+j
                        else:
                            line_names_corner = line_names_corner+"+"+j
                    if plot_MCMC:
                        corner.corner(
                            samples,
                            show_titles=True,
                            labels=labels_corner,
                            plot_datapoints=True,
                            quantiles=[0.16, 0.5, 0.84],
                        )
                        plt.suptitle(line_names_corner+"\n(normalized amplitude)", x=0.7)


                        self.parameter_time_series(initial, sampler, labels_corner)

                        self.quick_plot(x,y,model_name=copy_which_models[number], parlabels = labels_corner,sampler=sampler,best_fit_model=best_fit_model,theta_max=theta_max, normalization=local_normalization,
                                        hair_color="grey",title=self.target_name+" - "+line_names_corner, ylabel="F$_{\lambda} - F_{continuum}$ ["+f"{flux_density.unit}]",overplot_median_model=overplot_median_model, outputdir = self.output_dir)
                else:
                    continue
                






        redshifts = []
        for parameters in par_values_list:
            redshifts.append(parameters[0])
        redshifts = np.asarray(redshifts)
        average_redshift = np.average(redshifts)
        std_redshift = np.std(redshifts)
        
        if plot_spec:
            if overplot_archival_lines is not None:
                print("Archival lines are taken from NIST Atomic Spectra database: https://www.nist.gov/pml/atomic-spectra-database")
                print("We adopt vacuum wavelengths for lines with wavelengths < 2000 Angstroms and air wavelengths for lines with wavelengths > 2000 Angstroms.")
                print("If you use these lines in your research, please cite the NIST Atomic Spectra database appropriately.")
                archival_lines = np.loadtxt(str(libpath)+"/astro_lines.dat",dtype=str,delimiter=",")
                archival_wavelengths = archival_lines[:,0].astype(float)
                archival_wavelengths = archival_wavelengths*average_redshift + archival_wavelengths # correcting for the redshift
                archival_line_names = archival_lines[:,1]
                index = np.where((archival_wavelengths > wavelengths.value.min()) & (archival_wavelengths < wavelengths.value.max()))[0]  # Index to select the lines within our wavelength range
                archival_wavelengths = archival_wavelengths[index]
                archival_line_names = archival_line_names[index]
                index = []  # Index to select only the desired elements
                for n,element in enumerate(archival_line_names):
                    element_string = element.split("_")[0]
                    if element_string[0] == "[":
                        element_string = element_string[1:]
                    if element_string in overplot_archival_lines:
                        split_name = archival_line_names[n].split("_")
                        if len(split_name) == 3:
                            archival_line_names[n] = split_name[0]+" "+split_name[1]+fr"$\{split_name[2]}$"
                        else:
                            archival_line_names[n] = split_name[0]+split_name[1]
                        index.append(n)
                    
                archival_wavelengths = archival_wavelengths[index]
                archival_line_names = archival_line_names[index]

            plt.figure(figsize=(12,5))
            if overplot_archival_lines is not None:
                for number, line in enumerate(archival_wavelengths):
                    text_line_index = extraction.find_nearest(wavelengths.value, line)
                    if archival_line_names[number][0:2] == "H " or archival_line_names[number][0:2] == "He":
                        color = "C0"
                        step = 1.1
                    elif archival_line_names[number][0] == "O" or archival_line_names[number][0:2] == "[O":
                        color = "C2"
                        step = 1.22
                    elif archival_line_names[number][0] == "N" or archival_line_names[number][0:2] == "[N":
                        color = "C3"
                        step = 1.4
                    elif archival_line_names[number][0] == "S" or archival_line_names[number][0:2] == "[S":
                        color = "C4"
                        step = 1.6
                    else:
                        color = "black"
                        step = 1.1
                    plt.text(line, step*flux_density.value.max(), archival_line_names[number],rotation=90,fontsize=10,color=color, horizontalalignment="center", verticalalignment="bottom")
                    plt.vlines(line, flux_density.value[text_line_index], 0.98*step*flux_density.value.max(), color=color, linewidth=0.8,alpha=0.5)

            plt.plot(wavelengths,continuum_baseline,label="Continuum")
            plt.plot(wavelengths, flux_density, color='orange')
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(0,flux_density.value.max()*2)
            plt.ylabel("F$_{\lambda}$ "+f"[{flux_density.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.title(self.target_name+" - $z_{av} = $"+f"{round(average_redshift,7)}, $\sigma_z = {round(std_redshift,7)}$")
            plt.legend()
            plt.tight_layout()
            if save_results:
                plt.savefig(self.output_dir+f"/{self.target_name}_spec.pdf",bbox_inches='tight')

        if save_results:
            self.merge_fit_results(self.target_name,list_of_files=None,output_dir=self.output_dir)
            list_of_files = glob.glob(self.output_dir+"/*_line_fit_results.csv")
            for file in list_of_files:
                os.remove(file)
        
        if plot_spec or plot_MCMC:
            plt.show()

        return par_values_list, par_values_errors_list, par_names_list, samples_list, line_windows
    
    def line_dispersion_and_equiv_width(self, wavelengths_window, flux_density_window, continuum_baseline_window, line_std_deviation, line_name = None, plot = True):

        """
        
        This function estimates the line dispersion (aka the rms width of the line), the profile equivalent width, and the FWHM. All measures are independent
        on the line model (i.e. Gaussian, Lorentz, Voigt, or skewed models) and dependent on the interpolated line profile.

        Parameters
        ----------
        wavelengths_window: numpy.ndarray
            The wavelength window for the given line.
        flux_density_window: numpy.ndarray
            The flux density window for the given line.
        continuum_baseline_window: numpy.ndarray
            An array with the continuum density flux window for the given line. Standard easyspec units are in erg/cm2/s/A.
        line_std_deviation: numpy.ndarray (float)
            The standard deviation for the local continuum. This variable is an output of the function analysis.find_lines().
        line_name: string
            The line name.
        plot: boolean
            If True, the line and corresponding centroid and dispersion will be showed.
        
        Returns
        -------
        line_dispersion: float
            The line dispersion is well defined for arbitrary line profiles. See Eqs. 4 and 5 in Peterson et al. 2004 for some guidance.
            You should avoid using the line dispersion in case of blended lines. The result here is given in the observed frame.
        profile_equiv_width: float
            This is a model-independent estimate of the equivalent width and can be used with arbitrary line profiles.
            This is the integral of (Fc - F)/Fc over the wavelengths, where Fc is the continuum flux density and F is the interpolated flux density.
            The result is given in the observed frame.
        profile_FWHM: float
            The model-independent FWHM.
        line_disp_error: float
            The error on the line dispersion computed with a Monte Carlo simulation.
        equiv_width_error: float
            The error on the equivalent width computed with a Monte Carlo simulation.
        fwhm_error: float
            The error on the FWHM computed with a Monte Carlo simulation.
        """

        warnings.filterwarnings('ignore')
        
        def lambda_P(wavelengths_window,tck):
            return interpolate.splev(wavelengths_window, tck)*wavelengths_window
        
        def lambda2_P(wavelengths_window,tck):
            return interpolate.splev(wavelengths_window, tck)*(wavelengths_window-first_moment)**2
        
        def P(wavelengths_window,tck):
            return interpolate.splev(wavelengths_window, tck)
        
        def equiv_width(wavelengths_window,tck2):
            return interpolate.splev(wavelengths_window, tck2)

        tck = interpolate.splrep(wavelengths_window, (flux_density_window-continuum_baseline_window), k=1)
        integrand_EQW = (continuum_baseline_window - flux_density_window)/continuum_baseline_window
        tck2 = interpolate.splrep(wavelengths_window, integrand_EQW, k=1) 
        
        line_function = interpolate.splev(wavelengths_window, tck)
        line_function_positive = np.sqrt(line_function**2)  # We square the profile and take the squareroot such that we can also work with absorption lines
        line_peak = line_function_positive.max()
        index_peak = np.where(line_function_positive == line_peak)[0][0]
        integration_limit_0 = 0
        for n,flux_bin in enumerate(np.flip(line_function_positive[:index_peak])):
            if flux_bin > 2*line_std_deviation:
                if n < (index_peak-2):  # This condition is necessary such that we don't take e.g. wavelengths_window[-1] (negative index!)
                    try:
                        integration_limit_0 = wavelengths_window[index_peak-n-2]
                    except:
                        break
                else:
                    break
            else:
                break
        integration_limit_1 = 0
        for n,flux_bin in enumerate(line_function_positive[index_peak:]):
            if flux_bin > 2*line_std_deviation:
                try:
                    integration_limit_1 = wavelengths_window[index_peak+n+2]
                except:
                    break
            else:
                break
        numerator = quad(lambda_P, integration_limit_0, integration_limit_1,args=(tck,))[0]
        denominator = quad(P, integration_limit_0, integration_limit_1,args=(tck,))[0]

        first_moment = numerator/denominator
        
        second_moment = quad(lambda2_P, integration_limit_0, integration_limit_1,args=(tck,))[0]/denominator
        line_dispersion = np.sqrt(second_moment)

        profile_equiv_width = quad(equiv_width, integration_limit_0, integration_limit_1,args=(tck2,))[0]

        interpol_wavelenghts = np.linspace(integration_limit_0,integration_limit_1,1000)
        interpol_density_flux = interpolate.splev(interpol_wavelenghts, tck)
        if numerator > 0:
            line_peak_value = interpol_density_flux.max()
        else:
            line_peak_value = interpol_density_flux.min()
        max_index = extraction.find_nearest(interpol_density_flux, line_peak_value)
        max_index_for_error = extraction.find_nearest((flux_density_window-continuum_baseline_window), line_peak_value)  # This is used in the loop a few lines below.
        lambda_0_index = extraction.find_nearest(interpol_density_flux[0:max_index], line_peak_value/2)
        lambda_1_index = extraction.find_nearest(interpol_density_flux[max_index:], line_peak_value/2) + max_index
        profile_FWHM = np.abs(interpol_wavelenghts[lambda_1_index]-interpol_wavelenghts[lambda_0_index])
        

        # Error estimate:
        equiv_width_error = []
        line_disp_error = []
        fwhm_error = []
        median_profile = medfilt(flux_density_window-continuum_baseline_window, 7)
        std_line = np.std((median_profile-P(wavelengths_window,tck))[:int(len(median_profile)/4)] + (median_profile-P(wavelengths_window,tck))[-int(len(median_profile)/4):])
        for n in range(100):
            noise = np.random.uniform(size=len(flux_density_window))*std_line
            noisy_density_flux = flux_density_window + noise
            integrand_EQW = (continuum_baseline_window - noisy_density_flux)/continuum_baseline_window
            tck3 = interpolate.splrep(wavelengths_window, integrand_EQW, k=1)   
            local_EQW = quad(equiv_width, integration_limit_0, integration_limit_1,args=(tck3,))[0]
            equiv_width_error.append(local_EQW)

            tck4 = interpolate.splrep(wavelengths_window, (noisy_density_flux - continuum_baseline_window), k=1)
            line_disp_error.append(np.sqrt(quad(lambda2_P, integration_limit_0, integration_limit_1,args=(tck4,))[0]/denominator))

            lambda_0_index_error = extraction.find_nearest((noisy_density_flux[0:max_index_for_error]-continuum_baseline_window[0:max_index_for_error]), line_peak_value/2)
            lambda_1_index_error = extraction.find_nearest((noisy_density_flux[max_index_for_error:]-continuum_baseline_window[max_index_for_error:]), line_peak_value/2) + max_index_for_error
            fwhm_error.append(np.abs(wavelengths_window[lambda_1_index_error]-wavelengths_window[lambda_0_index_error]))

        equiv_width_error = np.std(np.asarray(equiv_width_error))
        line_disp_error = np.std(np.asarray(line_disp_error))
        fwhm_error = np.std(np.asarray(fwhm_error))

        if plot:
            plt.figure(figsize=(12,4))
            _, ax = plt.subplots(figsize=(12,4))
            plt.plot(wavelengths_window, interpolate.splev(wavelengths_window, tck), color = "C1")
            plt.plot(wavelengths_window,np.zeros(len(wavelengths_window)),color="black")
            plt.ylabel(r"Flux density - continuum [erg/cm$^2$/s/$\AA$]")
            plt.xlabel(r"Observed $\lambda$ [$\AA$]")
            plt.grid(which="both",linestyle="dotted")
            if numerator > 0:
                plt.vlines(first_moment,0,(flux_density_window-continuum_baseline_window).max(),colors="C0",linestyles=":",label="centroid")
                try:
                    plt.fill_betweenx([0,(flux_density_window-continuum_baseline_window).max()],first_moment-line_dispersion,first_moment+line_dispersion, color="C0", alpha=0.3, label=r"$\lambda_0 \pm \sigma_{rms}$")
                except:
                    print(f"Plotting error: first or second line moment is probably NaN. First moment: {first_moment}, second moment: {line_dispersion}")
                plt.plot([interpol_wavelenghts[lambda_0_index],interpol_wavelenghts[lambda_1_index]],[interpol_density_flux.max()/2,interpol_density_flux.max()/2],color="black")
                plt.text((interpol_wavelenghts[lambda_0_index]+interpol_wavelenghts[lambda_1_index])/2,1.05*interpol_density_flux.max()/2,"FWHM",horizontalalignment='center',)
            else:
                plt.vlines(first_moment,0,(flux_density_window-continuum_baseline_window).min(),colors="C0",linestyles=":",label="centroid")
                try:
                    plt.fill_betweenx([0,(flux_density_window-continuum_baseline_window).min()],first_moment-line_dispersion,first_moment+line_dispersion, color="C0", alpha=0.3, label=r"$\lambda_0 \pm \sigma_{rms}$")
                except:
                    print(f"Plotting error: first or second line moment is probably NaN. First moment: {first_moment}, second moment: {line_dispersion}")
                plt.plot([interpol_wavelenghts[lambda_0_index],interpol_wavelenghts[lambda_1_index]],[interpol_density_flux.min()/2,interpol_density_flux.min()/2],color="black")
                plt.text((interpol_wavelenghts[lambda_0_index]+interpol_wavelenghts[lambda_1_index])/2,0.90*interpol_density_flux.min()/2,"FWHM",horizontalalignment='center',)
            
            plt.scatter(interpol_wavelenghts[lambda_0_index],interpol_density_flux[lambda_0_index],color="black")
            plt.scatter(interpol_wavelenghts[lambda_1_index],interpol_density_flux[lambda_1_index],color="black")
            plt.text(0.05,0.8,"Not intended for\nblended lines!",color="red",transform = ax.transAxes)
            
            plt.legend()
            if line_name is not None:
                plt.title(line_name+" - Model-independent estimate for $\sigma_{rms}$ (observed frame)")


        return line_dispersion, profile_equiv_width, profile_FWHM, line_disp_error, equiv_width_error, fwhm_error


    def line_dispersion_skewed_models(self, par_values, theta_error, model="Lorentz"):

        """
        
        This function estimates the line dispersion (aka the rms width of the line) and the FWHM for the skwede line models.

        Parameters
        ----------
        par_values: list
            Array with the best-fit line parameters.
        theta_error: numpy.ndarray
            Array with the asymmetrical errors for each parameter.
        model: string
            The type of skewed function. Options are "Gauss" or "Lorentz".
        
        Returns
        -------
        line_dispersion: float
            The line dispersion is well defined for skewed line profiles. The result here is given in the observed frame.
        model_FWHM: float
            The skewed model FWHM.
        line_disp_error: float
            The error on the line dispersion computed with a Monte Carlo approach.
        fwhm_error: float
            The error on the FWHM computed with a Monte Carlo approach.
        """
        
        theta = np.asarray(par_values[1:])

        if model == "Lorentz":
            theta[2] = theta[2]/2
            theta_error[2] = theta_error[2]/2
            def lambda_P(wavelengths_window,theta):
                return model_skewed_lorentzian(theta,wavelengths_window)*wavelengths_window
            
            def lambda2_P(wavelengths_window,theta):
                return model_skewed_lorentzian(theta,wavelengths_window)*(wavelengths_window-first_moment)**2
            
            def Intensity(wavelengths_window,theta):
                return model_skewed_lorentzian(theta,wavelengths_window)
        else:
            theta[2] = theta[2]/(2*np.sqrt(2 * np.log(2))) # Gaussian component FWHM to sigma
            theta_error[2] = theta_error[2]/(2*np.sqrt(2 * np.log(2)))
            def lambda_P(wavelengths_window,theta):
                return model_skewed_gaussian(theta,wavelengths_window)*wavelengths_window
            
            def lambda2_P(wavelengths_window,theta):
                return model_skewed_gaussian(theta,wavelengths_window)*(wavelengths_window-first_moment)**2
            
            def Intensity(wavelengths_window,theta):
                return model_skewed_gaussian(theta,wavelengths_window)
        
        
        Normalization_factor_for_integration = 10**round(np.log10(theta[1]))
        theta[1] = theta[1]/Normalization_factor_for_integration

        x = np.linspace(theta[0]- 10*theta[2],theta[0]+ 10*theta[2])
        
        numerator = np.trapz(lambda_P(x,theta), x)*Normalization_factor_for_integration
        denominator = np.trapz(Intensity(x,theta), x)*Normalization_factor_for_integration

        first_moment = numerator/denominator

        second_moment = np.trapz(lambda2_P(x,theta), x)*Normalization_factor_for_integration/denominator
        line_dispersion = np.sqrt(second_moment)

        wavelength_range_0 = np.linspace(theta[0]-10*theta[2],theta[0],1000)
        wavelength_range_1 = np.linspace(theta[0],theta[0]+10*theta[2],1000)
        lambda_0_index = extraction.find_nearest(Intensity(wavelength_range_0,theta), theta[1]/2)
        lambda_1_index = extraction.find_nearest(Intensity(wavelength_range_1,theta), theta[1]/2)
        model_FWHM = np.abs(wavelength_range_1[lambda_1_index]-wavelength_range_0[lambda_0_index])

        # Error estimate:
        theta_error = np.mean(theta_error,axis=1)
        line_disp_error = []
        fwhm_error = []
        for n in range(100):
            noise = np.random.normal(scale=theta_error,size=len(theta))
            noisy_theta = theta+noise
            wavelength_range_0 = np.linspace(noisy_theta[0]-10*noisy_theta[2],noisy_theta[0],1000)
            wavelength_range_1 = np.linspace(noisy_theta[0],noisy_theta[0]+10*noisy_theta[2],1000)
            lambda_0_index = extraction.find_nearest(Intensity(wavelength_range_0,noisy_theta), noisy_theta[1]/2)
            lambda_1_index = extraction.find_nearest(Intensity(wavelength_range_1,noisy_theta), noisy_theta[1]/2)
            fwhm_error.append(np.abs(wavelength_range_1[lambda_1_index]-wavelength_range_0[lambda_0_index]))

            denominator = np.trapz(Intensity(x,noisy_theta), x)*Normalization_factor_for_integration
            second_moment = np.trapz(lambda2_P(x,noisy_theta), x)*Normalization_factor_for_integration/denominator
            line_disp_error.append(np.sqrt(second_moment))

        line_disp_error = np.nanstd(np.asarray(line_disp_error))
        fwhm_error = np.nanstd(np.asarray(fwhm_error))

        return line_dispersion, model_FWHM, line_disp_error, fwhm_error
    

    def error_propagation_voigt(self,FWHM_Lorentz,FWHM_Gauss,error_lorentz,error_gauss):

        """
        In this function we propagate the FWHM error for the Voigt model assuming independent variables.

        Parameters
        ----------
        FWHM_Lorentz: float
            The Lorentzian component FWHM.
        FWHM_Gauss: float
            The Gaussian component FWHM.
        error_lorentz: float
            The associated error to the Lorentzian FWHM.
        error_gauss: float
            The associated error to the Gaussian FWHM.
        
        Returns
        -------
        fwhm_error_voigt: float
            The propagated uncertainty in FWHM for the Voigt model.

        """

        fwhm_error_voigt = np.sqrt( (0.5346 + 0.5*0.2166*2*FWHM_Lorentz/np.sqrt(0.2166*FWHM_Lorentz**2 + FWHM_Gauss**2))*error_lorentz**2  + (FWHM_Gauss*error_gauss/np.sqrt(0.2166*FWHM_Lorentz**2 + FWHM_Gauss**2))**2 )

        return fwhm_error_voigt


    def equiv_width_error(self,integrand,line_window,line_parameters,line_par_errors, n_samples=100, n_points=300):

        """
        In this function, we estimate the modeled equivalent width error for a line based on a Monte Carlo simulation.

        Parameters
        ----------
        integrand: function
            The function to be integrated. See the details here: https://en.wikipedia.org/wiki/Equivalent_width
        line_window: list
            A list containing the wavelength limits around the line.
        line_parameters: list
            A list with the line best-fit parameters obtained with the MCMC method.
        line_par_errors: list
            A list with the asymmetrical parameter errors.
        n_samples: int
            Number of MC iterations.
        n_points: int
            Resolution of the wavelength axis.
        
        Returns
        -------
        EQW_error_down: float
            The equivalent width lower error for a modeled line.
        EQW_error_up: float
            The equivalent width upper error for a modeled line.
        """


        line_parameters = np.asarray(line_parameters)
        line_par_errors = np.asarray(line_par_errors)

        # Create wavelength grid
        lam = np.linspace(line_window[0], line_window[1], n_points)

        # Generate random samples for lower and upper error bounds
        rand = np.random.uniform(size=(n_samples, len(line_parameters)))
        params_down = line_parameters + rand * line_par_errors[:, 0]
        params_up   = line_parameters + rand * line_par_errors[:, 1]

        # Evaluate integrand and integrate using the trapezoidal rule
        eqw_down = np.array([
            np.trapz(integrand(lam, p), lam)
            for p in params_down
        ])
        eqw_up = np.array([
            np.trapz(integrand(lam, p), lam)
            for p in params_up
        ])

        # Return standard deviations as uncertainties
        EQW_error_down = float(np.std(eqw_down))
        EQW_error_up = float(np.std(eqw_up))
        return EQW_error_down, EQW_error_up
        

    def line_physics(self, wavelengths, flux_density, continuum_baseline, par_values_list, par_values_errors_list, par_names_list, line_windows, line_std_deviation, plot = True, save_file = True):

        """
        In this function, we compute several physical properties for the fitted lines.

        OBS 1: For the Voigt model, we assume independent variables when propagating the FWHM error. If this is not the case (you can check this in the MCMC
        corner plots), we recommend that you use the Gaussian or Lorentzian models.

        OBS 2: The integrated flux is computed by taking the equivalent width and multiplying it by the continuum value at the line center.

        OBS 3: The line dispersion is well defined for arbitrary line profiles. See e.g. Eqs. 4 and 5 in Peterson et al. 2004. This method is not recommended in case of blended lines.

        
        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        flux_density: numpy.ndarray (astropy.units erg/cm2/s/A)
            The calibrated spectrum in flux density.
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        par_values_list: list
            This is a list containing sublists with the best-fit values for each line model.
        par_values_errors_list: list
            A list with the asymmetrical errors for each parameter listed in par_values_list.
        par_names_list: list
            A list with the names of all the parameters used in each line model.
        line_windows: list
            A list containing the wavelength intervals around each line.
        line_std_deviation: numpy.ndarray (float)
            The standard deviation for the local continuum. This variable is an output of the function analysis.find_lines().
        plot: boolean
            If True, the line and corresponding centroid and dispersion will be showed.
        save_file: boolean
            If True, the results of this function will be saved in a .csv file in the output directory defined at analysis.load_calibrated_data().

        Returns
        -------
        profile_equiv_width_rest_frame: list (astropy.units Angstrom)
            A list with the equivalent widths (and respective errors) for each line and their respective errors in the rest frame. The values here are obtained with the
            interpolation of the line profile and are therefore independent on the line model (i.e. Gaussian, Lorentzian or Voigt).
        modeled_equiv_width_rest_frame: list (astropy.units Angstrom)
            A list with the model-dependent equivalent widths (and respective errors) for each line and their respective errors in the rest frame.
        profile_integrated_flux: list (astropy.units erg/cm2/s)
            A list with the model-independent line fluxes and their respective errors.
        modeled_integrated_flux: list (astropy.units erg/cm2/s)
            A list with the model-dependent line fluxes and their respective errors.
        profile_rest_frame_disp_velocity: list (astropy.units km/s)
            A list with the model-independent dispersion velocities and their respective errors in the rest frame.
        modeled_rest_frame_disp_velocity: list (astropy.units km/s)
            A list with the model-dependent dispersion velocities and their respective errors in the rest frame.
        profile_line_dispersion_rest_frame: list (astropy.units Angstrom)
            A list with the line dispersions (from the line second moments) and their respective errors in the rest frame.
        profile_rest_frame_fwhm: list (astropy.units Angstrom)
            A list with the model-independent line FWHMs and their respective errors in the rest frame.  
        modeled_rest_frame_fhwm: list (astropy.units Angstrom)
            A list with the model-dependent line FWHMs and their respective errors in the rest frame.        
        XXXXXXX_line_physics.csv
            Optional. This file contains all physics parameters for each modeled line and is saved in the output directory defined in the
            function analysis.load_calibrated_data(). "XXXXXXX" here stands for the target's name.
        """

        try:
            line_names = self.line_names
        except:
            line_names = [""]*len(par_values_list)

        modeled_equiv_width_rest_frame = []
        profile_equiv_width_rest_frame = []
        profile_line_dispersion_rest_frame = []
        modeled_integrated_flux = []
        profile_integrated_flux = []
        modeled_rest_frame_disp_velocity = []
        profile_rest_frame_disp_velocity = []
        modeled_rest_frame_fhwm = []
        profile_fwhm = []
        for number,line_parameters in enumerate(par_values_list):

            # taking the redshift and local continuum:
            z = line_parameters[0]
            peak_position = line_parameters[1]
            continuum_index = extraction.find_nearest(wavelengths.value, peak_position)

            # Computing the equivalent width directly from the data:
            window_indexes = np.where((wavelengths.value > line_windows[number][0]) & (wavelengths.value < line_windows[number][1]))[0]
            flux_density_window = flux_density[window_indexes]
            continuum_baseline_window = continuum_baseline[window_indexes]
            wavelengths_window = wavelengths[window_indexes]

            line_dispersion, profile_equiv_width, profile_FWHM, line_disp_error, equiv_width_error, fwhm_error = self.line_dispersion_and_equiv_width(wavelengths_window.value, flux_density_window.value, continuum_baseline_window, line_std_deviation = line_std_deviation[number], line_name = line_names[number], plot = plot)
            profile_disp_vel_error_down = c.to("km/s").value*np.sqrt(  line_disp_error*2/(peak_position**2) + (line_dispersion*par_values_errors_list[number][1][0]/(peak_position**2))**2 )
            profile_disp_vel_error_up = c.to("km/s").value*np.sqrt(  line_disp_error*2/(peak_position**2) + (line_dispersion*par_values_errors_list[number][1][1]/(peak_position**2))**2 )
            profile_rest_frame_disp_velocity.append([c.to("km/s").value*line_dispersion/peak_position, profile_disp_vel_error_down, profile_disp_vel_error_up])
            profile_integrated_flux.append([-profile_equiv_width*continuum_baseline[continuum_index], equiv_width_error*continuum_baseline[continuum_index], equiv_width_error*continuum_baseline[continuum_index]])
            line_dispersion_err_down = np.sqrt( (line_disp_error/(1+z))**2   +   (line_dispersion*par_values_errors_list[number][0][0]/((1+z)**2))**2   )
            line_dispersion_err_up = np.sqrt( (line_disp_error/(1+z))**2   +   (line_dispersion*par_values_errors_list[number][0][1]/((1+z)**2))**2   )
            line_dispersion = line_dispersion/(1+z)
            profile_equiv_width_err_down = np.sqrt( (equiv_width_error/(1+z))**2   +  (profile_equiv_width*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
            profile_equiv_width_err_up = np.sqrt( (equiv_width_error/(1+z))**2   +  (profile_equiv_width*par_values_errors_list[number][0][1]/((1+z)**2))**2      )
            profile_equiv_width = profile_equiv_width/(1+z)
            profile_FWHM_error_down = np.sqrt( (fwhm_error/(1+z))**2   +  (profile_FWHM*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
            profile_FWHM_error_up = np.sqrt( (fwhm_error/(1+z))**2   +  (profile_FWHM*par_values_errors_list[number][0][1]/((1+z)**2))**2      )
            profile_FWHM = profile_FWHM/(1+z)

            profile_equiv_width_rest_frame.append([profile_equiv_width,profile_equiv_width_err_down, profile_equiv_width_err_up])
            profile_line_dispersion_rest_frame.append([line_dispersion,line_dispersion_err_down, line_dispersion_err_up])
            profile_fwhm.append([profile_FWHM, profile_FWHM_error_down, profile_FWHM_error_up])
            
            # Computing the equivalent width from best-fit line models:
            if len(line_parameters) == 5:
                if par_names_list[number][4] == 'fwhm_Gauss':
                    # Voigt case:
                    def integrand(x,theta):
                        return -1*model_Voigt(theta,x)/continuum_baseline[continuum_index]  # The "-1" here is because our spectrum is already continuum-subtracted.
                    
                    equivalent_width = quad(integrand,line_windows[number][0],line_windows[number][1],args=line_parameters[1:])
                    EQW_error_down, EQW_error_up = self.equiv_width_error(integrand,line_windows[number],line_parameters[1:],par_values_errors_list[number][1:])
                    FWHM_Voigt = 0.5346*line_parameters[3] + np.sqrt(0.2166*line_parameters[3]**2 + line_parameters[4]**2)
                    disp_velocity = c.to("km/s").value*FWHM_Voigt/(peak_position*2*np.sqrt(2 * np.log(2)))
                    fwhm_error_down = self.error_propagation_voigt(par_values_list[number][3],par_values_list[number][4],par_values_errors_list[number][3][0],par_values_errors_list[number][4][0])
                    fwhm_error_up = self.error_propagation_voigt(par_values_list[number][3],par_values_list[number][4],par_values_errors_list[number][3][1],par_values_errors_list[number][4][1])
                    disp_velocity_err_down = c.to("km/s").value*fwhm_error_down/(peak_position*2*np.sqrt(2 * np.log(2)))
                    disp_velocity_err_up = c.to("km/s").value*fwhm_error_up/(peak_position*2*np.sqrt(2 * np.log(2)))
                    fwhm_error_down = np.sqrt( (fwhm_error_down/(1+z))**2   +  (FWHM_Voigt*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
                    fwhm_error_up = np.sqrt( (fwhm_error_up/(1+z))**2   +  (FWHM_Voigt*par_values_errors_list[number][0][1]/((1+z)**2))**2      )
                    fwhm_rest_frame = FWHM_Voigt/(1+z)
                elif par_names_list[number][3] == 'fwhm_Gauss':
                    # Skewed Gaussian case:
                    def integrand(x,theta):
                        return -1*model_skewed_gaussian(theta,x)/continuum_baseline[continuum_index]  # The "-1" here is because our spectrum is already continuum-subtracted.
                    
                    skewed_gaussian_parameters = line_parameters[1:]
                    skewed_gaussian_parameters[2] = skewed_gaussian_parameters[2]/(2*np.sqrt(2 * np.log(2)))  # This is necessary because the last Gaussian parameter saved in par_values_list is the FWHM and not the standard deviation.
                    equivalent_width = quad(integrand,line_windows[number][0],line_windows[number][1],args=skewed_gaussian_parameters)
                    skewed_gaussian_parameters_errors = np.asarray(par_values_errors_list[number][1:])
                    skewed_gaussian_parameters_errors[2] = skewed_gaussian_parameters_errors[2]/(2*np.sqrt(2 * np.log(2)))
                    EQW_error_down, EQW_error_up = self.equiv_width_error(integrand,line_windows[number],skewed_gaussian_parameters,skewed_gaussian_parameters_errors)
                
                    line_dispersion, model_FWHM, line_disp_error, fwhm_error = self.line_dispersion_skewed_models(par_values_list[number], skewed_gaussian_parameters_errors, model="Gauss")
                    
                    disp_velocity = c.to("km/s").value*line_dispersion/peak_position 
                    fwhm_rest_frame = model_FWHM/(1+z)
                    disp_velocity_err_down = c.to("km/s").value*line_disp_error/peak_position
                    disp_velocity_err_up = disp_velocity_err_down
                    fwhm_error_down = fwhm_error/(1+z)
                    fwhm_error_up = fwhm_error_down
                
                elif par_names_list[number][3] == 'fwhm_Lorentz':
                    # Skewed Lorentzian case:
                    def integrand(x,theta):
                        return -1*model_skewed_lorentzian(theta,x)/continuum_baseline[continuum_index]  # The "-1" here is because our spectrum is already continuum-subtracted.
                    
                    skewed_lorentzian_parameters = line_parameters[1:]
                    skewed_lorentzian_parameters[2] = skewed_lorentzian_parameters[2]/2  # This is necessary because the last Lorentzian parameter saved in par_values_list is the FWHM and not the HWHM.
                    equivalent_width = quad(integrand,line_windows[number][0],line_windows[number][1],args=skewed_lorentzian_parameters)
                    skewed_lorentzian_parameters_errors = np.asarray(par_values_errors_list[number][1:])
                    skewed_lorentzian_parameters_errors[2] = skewed_lorentzian_parameters_errors[2]/2
                    EQW_error_down, EQW_error_up = self.equiv_width_error(integrand,line_windows[number],skewed_lorentzian_parameters,skewed_lorentzian_parameters_errors)
                
                    line_dispersion, model_FWHM, line_disp_error, fwhm_error = self.line_dispersion_skewed_models(par_values_list[number], skewed_lorentzian_parameters_errors, model="Lorentz")
                    
                    disp_velocity = c.to("km/s").value*line_dispersion/peak_position 
                    fwhm_rest_frame = model_FWHM/(1+z)
                    disp_velocity_err_down = c.to("km/s").value*line_disp_error/peak_position
                    disp_velocity_err_up = disp_velocity_err_down
                    fwhm_error_down = fwhm_error/(1+z)
                    fwhm_error_up = fwhm_error_down

            elif par_names_list[number][3] == 'fwhm_Lorentz':
                # Lorentzian case:
                def integrand(x,theta):
                    return -1*model_Lorentz(theta,x)/continuum_baseline[continuum_index]  # The "-1" here is because our spectrum is already continuum-subtracted.
                
                equivalent_width = quad(integrand,line_windows[number][0],line_windows[number][1],args=line_parameters[1:])
                EQW_error_down, EQW_error_up = self.equiv_width_error(integrand,line_windows[number],line_parameters[1:],par_values_errors_list[number][1:])
                disp_velocity = c.to("km/s").value*line_parameters[3]/(2*peak_position) # The dispersion velocity for a Lorentzian is the half of its FWHM.
                disp_velocity_err_down = c.to("km/s").value*par_values_errors_list[number][3][0]/(peak_position*2)
                disp_velocity_err_up = c.to("km/s").value*par_values_errors_list[number][3][1]/(peak_position*2)
                fwhm_error_down = np.sqrt( (par_values_errors_list[number][3][0]/(1+z))**2   +  (line_parameters[3]*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
                fwhm_error_up = np.sqrt( (par_values_errors_list[number][3][1]/(1+z))**2   +  (line_parameters[3]*par_values_errors_list[number][0][1]/((1+z)**2))**2      )
                fwhm_rest_frame = line_parameters[3]/(1+z)
            else:
                # Gaussian case:
                def integrand(x,theta):
                    return -1*model_Gauss(theta,x)/continuum_baseline[continuum_index]  # The "-1" here is because our spectrum is already continuum-subtracted.
                
                gaussian_parameters = line_parameters[1:]
                gaussian_parameters[2] = gaussian_parameters[2]/(2*np.sqrt(2 * np.log(2)))  # This is necessary because the last Gaussian parameter saved in par_values_list is the FWHM and not the standard deviation.
                equivalent_width = quad(integrand,line_windows[number][0],line_windows[number][1],args=gaussian_parameters)
                gaussian_parameters_errors = par_values_errors_list[number][1:]
                gaussian_parameters_errors[2] = gaussian_parameters_errors[2]/(2*np.sqrt(2 * np.log(2)))
                EQW_error_down, EQW_error_up = self.equiv_width_error(integrand,line_windows[number],gaussian_parameters,gaussian_parameters_errors)
                disp_velocity = c.to("km/s").value*line_parameters[3]/(peak_position*2*np.sqrt(2 * np.log(2))) # The dispersion velocity for a Gaussian is its standard deviation given in velocity units.
                disp_velocity_err_down = c.to("km/s").value*par_values_errors_list[number][3][0]/(peak_position*2*np.sqrt(2 * np.log(2)))
                disp_velocity_err_up = c.to("km/s").value*par_values_errors_list[number][3][1]/(peak_position*2*np.sqrt(2 * np.log(2)))
                fwhm_error_down = np.sqrt( (par_values_errors_list[number][3][0]/(1+z))**2   +  (line_parameters[3]*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
                fwhm_error_up = np.sqrt( (par_values_errors_list[number][3][1]/(1+z))**2   +  (line_parameters[3]*par_values_errors_list[number][0][1]/((1+z)**2))**2      )
                fwhm_rest_frame = line_parameters[3]/(1+z)

            modeled_integrated_flux.append([-equivalent_width[0]*continuum_baseline[continuum_index], EQW_error_down*continuum_baseline[continuum_index] , EQW_error_up*continuum_baseline[continuum_index]])
            EQW_error_down = np.sqrt( (EQW_error_down/(1+z))**2   +  (equivalent_width[0]*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
            EQW_error_up = np.sqrt( (EQW_error_up/(1+z))**2   +  (equivalent_width[0]*par_values_errors_list[number][0][0]/((1+z)**2))**2      )
            modeled_rest_frame_disp_velocity.append([disp_velocity,disp_velocity_err_down,disp_velocity_err_up])  
            modeled_rest_frame_fhwm.append([fwhm_rest_frame,fwhm_error_down,fwhm_error_up])
            modeled_equiv_width_rest_frame.append([equivalent_width[0]/(1+z), EQW_error_down, EQW_error_up]) # We divide the measured equivalent width by 1+z to get the rest-frame equiv width.
            

        modeled_rest_frame_fhwm = modeled_rest_frame_fhwm*u.AA
        modeled_rest_frame_disp_velocity = modeled_rest_frame_disp_velocity*u.km/u.s
        profile_rest_frame_disp_velocity = profile_rest_frame_disp_velocity*u.km/u.s
        profile_equiv_width_rest_frame = profile_equiv_width_rest_frame*u.AA
        profile_line_dispersion_rest_frame = profile_line_dispersion_rest_frame*u.AA
        modeled_equiv_width_rest_frame = modeled_equiv_width_rest_frame*u.AA
        modeled_integrated_flux = modeled_integrated_flux*u.erg/u.cm**2/u.s
        profile_integrated_flux = profile_integrated_flux*u.erg/u.cm**2/u.s
        profile_rest_frame_fwhm = profile_fwhm*u.AA

        if plot:
            plt.show()

        if save_file:
            f = open(f'{self.output_dir}/'+self.target_name+"_line_physics.csv","w")
            f.write("Line name, profile_equiv_width_rest_frame [Angstrom], err_down, err_up,")
            f.write("modeled_equiv_width_rest_frame [Angstrom], err_down, err_up,")
            f.write("profile_integrated_flux [erg/cm2/s], err_down, err_up,")
            f.write("modeled_integrated_flux [erg/cm2/s], err_down, err_up,")
            f.write("profile_rest_frame_disp_velocity [km/s], err_down, err_up,")
            f.write("modeled_rest_frame_disp_velocity [km/s], err_down, err_up,")
            f.write("profile_line_dispersion_rest_frame [Angstrom], err_down, err_up,")
            f.write("profile_rest_frame_fwhm [Angstrom], err_down, err_up,")
            f.write("modeled_rest_frame_fhwm [Angstrom], err_down, err_up\n")
            for number in range(len(profile_equiv_width_rest_frame)):
                f.write(self.line_names[number])
                f.write(","+str(profile_equiv_width_rest_frame[number][0].value)+","+str(profile_equiv_width_rest_frame[number][1].value)+","+str(profile_equiv_width_rest_frame[number][2].value))
                f.write(","+str(modeled_equiv_width_rest_frame[number][0].value)+","+str(modeled_equiv_width_rest_frame[number][1].value)+","+str(modeled_equiv_width_rest_frame[number][2].value))
                f.write(","+str(profile_integrated_flux[number][0].value)+","+str(profile_integrated_flux[number][1].value)+","+str(profile_integrated_flux[number][2].value))
                f.write(","+str(modeled_integrated_flux[number][0].value)+","+str(modeled_integrated_flux[number][1].value)+","+str(modeled_integrated_flux[number][2].value))
                f.write(","+str(profile_rest_frame_disp_velocity[number][0].value)+","+str(profile_rest_frame_disp_velocity[number][1].value)+","+str(profile_rest_frame_disp_velocity[number][2].value))
                f.write(","+str(modeled_rest_frame_disp_velocity[number][0].value)+","+str(modeled_rest_frame_disp_velocity[number][1].value)+","+str(modeled_rest_frame_disp_velocity[number][2].value))
                f.write(","+str(profile_line_dispersion_rest_frame[number][0].value)+","+str(profile_line_dispersion_rest_frame[number][1].value)+","+str(profile_line_dispersion_rest_frame[number][2].value))
                f.write(","+str(profile_rest_frame_fwhm[number][0].value)+","+str(profile_rest_frame_fwhm[number][1].value)+","+str(profile_rest_frame_fwhm[number][2].value))
                f.write(","+str(modeled_rest_frame_fhwm[number][0].value)+","+str(modeled_rest_frame_fhwm[number][1].value)+","+str(modeled_rest_frame_fhwm[number][2].value))
                f.write("\n")
            f.close()
        
        
        return profile_equiv_width_rest_frame, modeled_equiv_width_rest_frame, profile_integrated_flux, modeled_integrated_flux, profile_rest_frame_disp_velocity, modeled_rest_frame_disp_velocity, profile_line_dispersion_rest_frame, profile_rest_frame_fwhm, modeled_rest_frame_fhwm


    def BH_mass_Hbeta_VP2006(self, wavelengths, continuum_baseline, FWHM_Hbeta, par_values_Hbeta, integrated_flux_Hbeta, H0=70):

        """
        This function estimates the black hole mass based on Vestergaard & Peterson, 2006, ApJ, 641, "DETERMINING CENTRAL BLACK HOLE MASSES IN DISTANT ACTIVE
        GALAXIES AND QUASARS. II. IMPROVED OPTICAL AND UV SCALING RELATIONSHIPS". As stated in this work, here we assume a cosmology with H0 = 70 km/s/Mpc, Omega_Lambda = 0.7,
        and Omega_matter = 0.3, although we allow the user to change the value of the Hubble constant (H0).

        We assume a systematic error of 0.43 dex for the estimated black hole masses. This value is reported in Vestergaard & Peterson, 2006. Since this error is
        much higher than the errors in FWHM and integrated flux, we simply ignore the measured errors in these parameters.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        FWHM_Hbeta: float (astropy.units Angstrom)
            The FWHM for the Hbeta line in Angstrom units.
        par_values_Hbeta: list
            The list with the best fit values for the Hbeta line. This information is contained in the variable "par_values_list" returned from the function
            analysis.fit_lines().
        integrated_flux_Hbeta: float (astropy.units Angstrom)
            The integrated flux for the Hbeta line in erg/cm2/s units.
        H0: float
            This is the Hubble constant value. Default is 70 km/s/Mpc.

        Returns
        -------
        log10_BH_mass_continuum: float
            The black hole mass in log10 scale and its corresponding error in dex computed based on Eq. 5 from Vestergaard & Peterson, 2006, ApJ, 641.
        log10_BH_mass_line_lum: float
            The black hole mass in log10 scale and its corresponding error in dex computed based on Eq. 6 from Vestergaard & Peterson, 2006, ApJ, 641.
        """

        systematic_error = 0.43 # dex. Result from Vestergaard & Peterson, 2006.
        z = par_values_Hbeta[0]
        FWHM_Hbeta_velocity = c.to("km/s").value*FWHM_Hbeta.value*(1+z)/par_values_Hbeta[1]

        cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0 = 2.7) # Omega lambda is implicitly 0.7
        Distance = cosmo.luminosity_distance(z).value*3.086e24  # Luminosity distance. The constant converts Mpc to cm.
        Lum_Hbeta = integrated_flux_Hbeta.value*4*np.pi*(Distance**2)

        continuum_index = extraction.find_nearest(wavelengths.value, 5100*(1+z))
        Lum_5100 = continuum_baseline[continuum_index]*wavelengths.value[continuum_index]*4*np.pi*(Distance**2)

        log10_BH_mass_continuum = np.log10( ((FWHM_Hbeta_velocity/1000)**2) * np.sqrt((Lum_5100/(10**44))) ) + 6.91

        log10_BH_mass_line_lum = np.log10(  ((FWHM_Hbeta_velocity/1000)**2) * ((Lum_Hbeta/(10**42)))**0.63 ) + 6.67

        log10_BH_mass_continuum = [log10_BH_mass_continuum,systematic_error]
        log10_BH_mass_line_lum = [log10_BH_mass_line_lum,systematic_error]
        return log10_BH_mass_continuum, log10_BH_mass_line_lum


    def BH_mass_CIV_VP2006(self, wavelengths, continuum_baseline, FWHM_CIV, par_values_CIV, H0=70):

        """
        This function estimates the black hole mass based on Vestergaard & Peterson, 2006, ApJ, 641, "DETERMINING CENTRAL BLACK HOLE
        MASSES IN DISTANT ACTIVE GALAXIES AND QUASARS. II. IMPROVED OPTICAL AND UV SCALING RELATIONSHIPS". As stated in this work,
        here we assume a cosmology with H0 = 70 km/s/Mpc, Omega_Lambda = 0.7, and Omega_matter = 0.3, although we allow the user to
        change the value of the Hubble constant (H0).

        We assume a systematic error of 0.36 dex for the estimated black hole masses. This value is reported in Vestergaard & Peterson, 2006. Since this error is
        much higher than the errors in FWHM and integrated flux, we simply ignore the measured errors in these parameters.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        FWHM_CIV: float (astropy.units Angstrom)
            The FWHM for the CIV line in Angstrom units.
        par_values_CIV: list
            The list with the best fit values for the CIV line. This information is contained in the variable "par_values_list" returned from the function
            analysis.fit_lines().
        H0: float
            This is the Hubble constant value. Default is 70 km/s/Mpc.

        Returns
        -------
        log10_BH_mass_CIV: float
            The black hole mass in log10 scale and its corresponding error in dex computed based on Eq. 8 from Vestergaard & Peterson, 2006, ApJ, 641.
        """

        systematic_error_FWHM = 0.36 # dex. Result from Vestergaard & Peterson, 2006.
        z = par_values_CIV[0]
        FWHM_CIV_velocity = c.to("km/s").value*FWHM_CIV.value*(1+z)/par_values_CIV[1]

        cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0 = 2.7) # Omega lambda is implicitly 0.7
        Distance = cosmo.luminosity_distance(z).value*3.086e24  # Luminosity distance. The constant converts Mpc to cm.

        continuum_index = extraction.find_nearest(wavelengths.value, 1350*(1+z))
        Lum_1350 = continuum_baseline[continuum_index]*wavelengths.value[continuum_index]*4*np.pi*(Distance**2)

        log10_BH_mass_CIV = np.log10( ((FWHM_CIV_velocity/1000)**2) * (Lum_1350/(10**44))**0.53 ) + 6.66

        log10_BH_mass_CIV = [log10_BH_mass_CIV,systematic_error_FWHM]
        return log10_BH_mass_CIV


    def BH_mass_MgII_VO2009(self, wavelengths, continuum_baseline, FWHM_MgII, par_values_MgII, H0=70):

        """
        This function estimates the black hole mass based on Vestergaard & Osmer, 2009, ApJ, 699, "MASS FUNCTIONS OF THE ACTIVE BLACK HOLES
        IN DISTANT QUASARS FROM THE LARGE BRIGHT QUASAR SURVEY, THE BRIGHT QUASAR SURVEY, AND THE COLOR-SELECTED SAMPLE OF THE SDSS FALL
        EQUATORIAL STRIPE". As stated in this work, here we assume a cosmology with H0 = 70 km/s/Mpc, Omega_Lambda = 0.7,
        and Omega_matter = 0.3, although we allow the user to change the value of the Hubble constant (H0).

        We assume a systematic error of 0.55 dex for the estimated black hole masses. This value is reported in Vestergaard & Osmer, 2009.
        Since this error is much higher than the errors in FWHM and integrated flux, we simply ignore the measured errors in these parameters.

        Parameters
        ----------
        wavelengths: numpy.ndarray (astropy.units Angstrom)
            The wavelength solution for the given spectrum.
        continuum_baseline: numpy.ndarray (float)
            An array with the continuum density flux. Standard easyspec units are in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        FWHM_MgII: float (astropy.units Angstrom)
            The FWHM for the MgII line in Angstrom units.
        par_values_MgII: list
            The list with the best fit values for the MgII line. This information is contained in the variable "par_values_list" returned from the function
            analysis.fit_lines().
        H0: float
            This is the Hubble constant value. Default is 70 km/s/Mpc.

        Returns
        -------
        log10_BH_mass_MgII: float
            The black hole mass in log10 scale and its corresponding error in dex computed based on Eq. 1 from Vestergaard & Osmer, 2009, ApJ, 699,
            taking the continuum luminosity at 3000 Angstroms. If the continuum at 3000 Angstroms is not available, it automatically uses the
            continuum at 2100 Angstroms.
        """

        systematic_error_FWHM = 0.55 # dex. Result from Vestergaard & Osmer, 2009.
        z = par_values_MgII[0]
        FWHM_MgII_velocity = c.to("km/s").value*FWHM_MgII.value*(1+z)/par_values_MgII[1]

        cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0 = 2.7) # Omega lambda is implicitly 0.7
        Distance = cosmo.luminosity_distance(z).value*3.086e24  # Luminosity distance. The constant converts Mpc to cm.

        continuum_index = extraction.find_nearest(wavelengths.value, 3000*(1+z))
        if wavelengths.value[continuum_index] < 0.9*3000*(1+z):
            continuum_index = extraction.find_nearest(wavelengths.value, 2100*(1+z))
            Lum_2100 = continuum_baseline[continuum_index]*wavelengths.value[continuum_index]*4*np.pi*(Distance**2)
            log10_BH_mass_MgII = np.log10( ((FWHM_MgII_velocity/1000)**2) * (Lum_2100/(10**44))**0.5 ) + 6.79
        else:
            Lum_3000 = continuum_baseline[continuum_index]*wavelengths.value[continuum_index]*4*np.pi*(Distance**2)
            log10_BH_mass_MgII = np.log10( ((FWHM_MgII_velocity/1000)**2) * (Lum_3000/(10**44))**0.5 ) + 6.86

        log10_BH_mass_MgII = [log10_BH_mass_MgII,systematic_error_FWHM]
        return log10_BH_mass_MgII

    def BH_mass_Halpha_Shen2011(self, FWHM_Halpha, par_values_Halpha, integrated_flux_Halpha, H0=70):

        """
        This function estimates the black hole mass based on Shen at al. 2011, ApJS, 194:45, "A CATALOG OF QUASAR PROPERTIES FROM
        SLOAN DIGITAL SKY SURVEY DATA RELEASE 7". As stated in this work, here we assume a cosmology with H0 = 70 km/s/Mpc, Omega_Lambda = 0.7,
        and Omega_matter = 0.3, although we allow the user to change the value of the Hubble constant (H0).

        We assume a systematic error of 0.18 dex for the estimated black hole masses. This value is reported in Shen et al. 2011.
        Since this error is much higher than the errors in FWHM and integrated flux, we simply ignore the measured errors in these parameters.

        Parameters
        ----------
        FWHM_Halpha: float (astropy.units Angstrom)
            The FWHM for the Halpha line in Angstrom units.
        par_values_Halpha: list
            The list with the best fit values for the Halpha line. This information is contained in the variable "par_values_list" returned from the function
            analysis.fit_lines().
        integrated_flux_Halpha: float (astropy.units Angstrom)
            The integrated flux for the Halpha line in erg/cm2/s units.
        H0: float
            This is the Hubble constant value. Default is 70 km/s/Mpc.

        Returns
        -------
        log10_BH_mass_Halpha: float
            The black hole mass in log10 scale and its corresponding error in dex computed based on Eq. 10 from Shen et al. 2011. 
        """

        systematic_error_FWHM = 0.18
        z = par_values_Halpha[0]
        FWHM_Halpha_velocity = c.to("km/s").value*FWHM_Halpha.value*(1+z)/par_values_Halpha[1]

        cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0 = 2.7) # Omega lambda is implicitly 0.7
        Distance = cosmo.luminosity_distance(z).value*3.086e24  # Luminosity distance. The constant converts Mpc to cm.
        Lum_Halpha = integrated_flux_Halpha.value*4*np.pi*(Distance**2)

        log10_BH_mass_Halpha = np.log10(  (FWHM_Halpha_velocity**2.1) * ((Lum_Halpha/(10**42)))**0.43 ) + 0.379
        
        log10_BH_mass_Halpha = [log10_BH_mass_Halpha,systematic_error_FWHM]
        return log10_BH_mass_Halpha


    def get_highest_resolution(self, specs_dict): 

        """ 
        This function gets the spectrum with the highest resolution in specs_dict.

        Parameters:
        -----------
        specs_dict: dictionary
            A dictionary where each key is a spectrum name and each value is a list containing [lambda0, lambdaf, R].

        Returns:
        --------
        highest_R_spec: string
            The name of the spectrum with the highest resolution.
        highest_R: float
            The highest resolution value found in specs_dict.
        """
        highest_R = 0
        highest_R_spec = None

        for spec in specs_dict: 

            resolution = specs_dict[spec][2]  # retrieves the list for the current 'spec' and gets the element at index [2], which is 'R'

            if resolution > highest_R:
                highest_R = resolution
                highest_R_spec = spec
        return highest_R_spec, highest_R
    

    def create_common_grid(self, specs_dict, over_resolution_factor):

        """ 
        This function creates a common wavelength grid for all spectra to be used for the final stacked spectrum.

        Parameters:
        -----------
        specs_dict: dictionary
            A dictionary where each key is a spectrum and each value is a list containing [lambda0, lambdaf, R].
        over_resolution_factor: int
            This factor increases the number of points in the stacked spectrum wavelength grid. This value must be greater than 1.

        Returns:
        --------
        wavelengths_stacked: numpy.ndarray
            An array containing the interpolated wavelength values for the stacked spectrum.
        """

        highest_R = self.get_highest_resolution(specs_dict)[1]  # stores the 2° value of the return of the function(the resolution) in "highest_R"
        all_lambda0 = [value[0] for value in specs_dict.values()]   # stores lambda0 for all spectra (spect_dict values are [lambda0, lambdaf, R])
        all_lambdaf = [value[1] for value in specs_dict.values()]   # stores lambdaf for all spectra

        lambda_min = min(all_lambda0)  # smallest value of lambda0 among all spectra
        lambda_max = max(all_lambdaf)  # largest value of lambdaf among all spectra
        N_points = int(over_resolution_factor * highest_R * (lambda_max - lambda_min))   # number of points for the new wavelength array, using 5x highest_R

        wavelengths_stacked = np.linspace(lambda_min,lambda_max,N_points)  # array with the wavelength values for the stacked spectrum

        return wavelengths_stacked
    

    def interpolate_spectra(self, list_of_wavelengths, list_of_fluxes, wavelengths_stacked):

        """ 
        This function interpolates all spectra to the common wavelength grid of the stacked spectrum,
        fills the grid with NaN's where there is no data and creates a matrix where each row is an interpolated spectrum.

        Parameters:
        ----------- 
        list_of_wavelengths: list
            A list where each element is a numpy array containing the wavelength values of each spectrum.
        list_of_fluxes: list
            A list where each element is a numpy array containing the flux values of each spectrum.
        wavelengths_stacked: numpy.ndarray
            An array containing the wavelength values of the stacked spectrum in Å. These are not the measured wavelengths but the values of the common grid created for stacking.

        Returns:
        --------  
        spectra_interp_matrix: numpy.ndarray
            A matrix where each row is an interpolated spectrum with the size of the common wavelength grid.
        """
        spectra_interp_list = []

        for wavelength, flux in zip(list_of_wavelengths, list_of_fluxes):
            
            # Creates the splines and interpolates
            tck = interpolate.splrep(wavelength, flux, k=3)
            flux_interp = interpolate.splev(wavelengths_stacked, tck)
            
            # Creates an array of NaN's with the same shape as wavelengths_stacked
            filled_flux = np.full_like(wavelengths_stacked, np.nan, dtype=float)

            # Creates a mask for the valid wavelength range of each spectrum
            mask = (wavelengths_stacked >= wavelength.min()) & (wavelengths_stacked <= wavelength.max())
            
            # Fills the valid range in filled_flux with the interpolated flux values
            filled_flux[mask] = flux_interp[mask]
            
            spectra_interp_list.append(filled_flux)

        # Creates the matrix where each row is an interpolated spectrum
        spectra_interp_matrix = np.vstack(spectra_interp_list)

        return spectra_interp_matrix


    def calculate_stack(self, spectra_interp_matrix, method):
    
        """
        This function calculates the stacked spectrum using the specified method (median or mean).

        Parameters:
        -----------
        spectra_interp_matrix: numpy.ndarray
            A matrix where each row is an interpolated spectrum with the size of the common wavelength grid.
        method: string
            The stacking method to use, either "median" or "mean".

        Returns:
        --------  
        stacked_flux: numpy.ndarray
            An array containing the interpolated flux values of the stacked spectrum. 
        """
        
        if method == "median":
            stacked_flux = np.nanmedian(spectra_interp_matrix, axis=0)

        elif method == "mean":
            stacked_flux = np.nanmean(spectra_interp_matrix, axis=0)

        else:
            raise ValueError("Invalid stacking method. Use 'median' or 'mean'.")
            
        return stacked_flux
    

    def stack_calib_spectra(self, input_data, method="median", target_name=None, output_dir="./", over_resolution_factor=5, save_file=False, plot=True, plot_overlayed_spectra=True):
    
        """
        This function stacks calibrated .dat spectra from a list of file paths or a directory containing .dat spectra.

        Parameters: 
        -----------
        input_data: list or string
            A list of ".dat" file paths containing calibrated spectra or a directory path containing ".dat" spectra.
        method: string
            The stacking method to use, either "median" or "mean".
        target_name: string
            Optional. This will be the title of the plot.
        output_dir: string
            The directory where the stacked spectrum file will be saved if save_file is True.
        over_resolution_factor: int
            This factor increases the number of points in the stacked spectrum wavelength grid. This value must be greater than 1. Default value = 5.
        save_file: bool
            If True, saves the stacked spectrum to a .dat file.
        plot: bool
            If True, plots the stacked spectrum.
        plot_overlayed_spectra: bool
            If True, plots all individual spectra overlaid with the stacked spectrum.

        Returns:
        --------  
        wavelengths_stacked: numpy.ndarray (astropy.units Å)
            An array containing the wavelength values of the stacked spectrum in Å. These are not the measured wavelengths but the values of the common grid created for stacking.
        stacked_flux: numpy.ndarray (astropy.units. erg / (Å cm² s))
            An array containing the interpolated flux values of the stacked spectrum in erg / (Å cm² s). 
        """
        files_list = []
        
        if isinstance(input_data, list):   # Checks if input_data is a list
            
            for file_path in input_data:
                if not file_path.endswith(".dat"):   # Test if all elements are .dat files
                    raise RuntimeError("The input_data variable must be a list of data paths or a directory filled with .dat spectra.")
        
            files_list = input_data

        elif isinstance(input_data,str) and Path(input_data).is_dir():  # Checks if input_data is a directory path
            
            files_list = [str(path) for path in Path(input_data).glob("*.dat")]
            
        else:
            raise RuntimeError("The input_data variable must be a list of data paths or a directory filled with .dat spectra.")

        list_of_wavelengths = []
        list_of_fluxes = []

        for file in files_list:
                wavelength, flux = self.load_calibrated_data(file, target_name=target_name, output_dir="./",plot=False)
                list_of_wavelengths.append(wavelength.value)
                list_of_fluxes.append(flux.value)
        
        names = [Path(file_path).stem for file_path in files_list]   # fills the names list with the name of each file(.stem removes the .dat extension)

        lambda0_list = []
        lambdaf_list = []
        specs_dict = {}

        for name, wavelength in zip(names, list_of_wavelengths): # iterates through both lists simultaneously
            lambda0 = wavelength[0]       # first wavelength value
            lambdaf = wavelength[-1]      # last wavelength value
            lambda0_list.append(lambda0)  # adds all lambda0 values to lambda0_list
            lambdaf_list.append(lambdaf)  # adds all lambdaf values to lambdaf_list
            delta_lambda = lambdaf - lambda0    
            resolution = len(wavelength)/(delta_lambda)  # points per wavelength

            specs_dict[name] = [lambda0, lambdaf, resolution]  # assigns each name to its corresponding values to fill the dictionary

        wavelengths_stacked = self.create_common_grid(specs_dict, over_resolution_factor)   # Creates the wavelength interval for the stacked spectrum
        spectra_interp_matrix = self.interpolate_spectra(list_of_wavelengths, list_of_fluxes, wavelengths_stacked)  # Interpolates all spectra to the common wavelength grid
        stacked_flux = self.calculate_stack(spectra_interp_matrix, method)  # Calculates the stacked flux
        
        if plot and not plot_overlayed_spectra:   

            plt.figure(figsize=(12, 4), dpi=150)
            plt.plot(wavelengths_stacked, stacked_flux, color='orange', linewidth=0.5)
            plt.xlabel('Observed ${\lambda}$[$\AA$]')
            plt.ylabel(r'$F_{\lambda}$ [erg / ($\AA$ cm² s)]', fontsize=12)
            if target_name is not None:
                plt.title('Stacked Spectrum of '+ target_name, fontsize=12)
            else:
                plt.title('Stacked Spectrum', fontsize=12)
            plt.grid(True, which='both', linewidth=0.1, linestyle='--', color='gray')
            plt.minorticks_on()
            plt.legend()
            plt.show()

        elif plot and plot_overlayed_spectra:   # Plots all individual spectra overlaid with the stacked spectrum
            
            plt.figure(figsize=(12, 4), dpi=150)

            for i in range(spectra_interp_matrix.shape[0]):
                plt.plot(wavelengths_stacked, spectra_interp_matrix[i], alpha=0.8, linewidth=0.5)
            
            plt.plot(wavelengths_stacked, stacked_flux, label='Stacked Spectrum', color='black', linewidth=0.5)
            plt.xlabel('Observed ${\lambda}$[$\AA$]')
            plt.ylabel(r'$F_{\lambda}$ [erg / ($\AA$ cm² s)]', fontsize=12)
            if target_name is not None:
                plt.title('Stacked Spectrum of '+ target_name + ' overlayed with original spectra', fontsize=12)
            else:
                plt.title('Stacked Spectrum overlayed with original spectra', fontsize=12)
            plt.grid(True, which='both', linewidth=0.1, linestyle='--', color='gray')
            plt.minorticks_on()
            plt.legend()
            plt.show()

        if save_file:
            output_path = f"{str(Path(output_dir))}/stacked_spectrum_{method}.dat"
            np.savetxt(output_path, np.column_stack((wavelengths_stacked, stacked_flux)))

        return wavelengths_stacked*u.AA, stacked_flux*u.erg/(u.AA*u.cm**2*u.s)