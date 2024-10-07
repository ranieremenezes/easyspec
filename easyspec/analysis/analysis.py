import numpy as np
from math import factorial
import scipy
from astropy.modeling.models import Gaussian1D, Voigt1D, Lorentz1D
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import stats
import glob
from pathlib import Path
from multiprocessing import Pool
import time
import warnings
from matplotlib.ticker import AutoMinorLocator
from scipy import interpolate
from easyspec.extraction import extraction
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter
from astropy import units as u
from astropy.modeling.models import Linear1D
from scipy.signal import medfilt
import platform
import os

OS_name = platform.system()
plt.rcParams.update({'font.size': 12})
libpath = Path(__file__).parent.resolve() / Path("lines")
extraction = extraction()


easyspec_analysis_version = "1.0.0"


class analysis:

    """This class contains all the functions necessary to perform the analysis of calibrated spectral data."""

    def __init__(self): 
        # Print the current version of easyspec-extraction       
        print("easyspec-analysis version: ",easyspec_analysis_version)
    
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
        wavelengths, flux_density, wavelength_systematic_error = data[:,0]*u.angstrom, data[:,1]*u.erg / u.cm**2 / u.s / u.AA, data[:,2][0]*u.angstrom  # Angstrom, erg/cm2/s/Angstrom, Angstrom

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

        self.wavelength_systematic_error = wavelength_systematic_error

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
            The continuum will be computed only within these wavelength regions and then extrapolated everywhere else. E.g.: continuum_regions = [[3000,6830],[6930,7500]].
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
            plt.title("The continuum noise is independently estimated for each one of these regions")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())

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
    



    def all_models(self, model_name, custom_function=None):
        """
        This function is used to select a specific line model. E.g.: Gaussian, Lorentz, doubleLorentz, and so on.

        Parameters
        ----------
        model_name: string
            The name of the line model, e.g. "Gaussian".
        custom_function: method
            Optional. You can input a custom model here.

        Returns
        -------
        models[model_name]: method
            returns a function with the desired line model.
        """
        
        models = {"Gaussian" : self.model_Gauss, "Lorentz" : self.model_Lorentz, "Voigt" : self.model_Voigt, "doubleVoigt" : self.model_double_Voigt, "tripleVoigt": self.model_triple_Voigt, "custom" : custom_function}
        if model_name not in models.keys():
            raise Exception("Invalid model_name. Options are: Gaussian, Lorentz, Voigt, doubleVoigt, tripleVoigt, custom")
        return models[model_name]

    def model_double_Voigt(self, theta,x):
        x_0, amplitude_L, fwhm_G, fwhm_L, x_2, amplitude2_L, fwhm2_G, fwhm2_L = theta
        a = Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G) + Voigt1D(x_2, amplitude2_L, fwhm2_L, fwhm2_G)
        return a(x)

    def model_triple_Voigt(self, theta,x):
        x_0, amplitude_L, fwhm_G, fwhm_L, x_2, amplitude2_L, fwhm2_G, fwhm2_L, x_3, amplitude3_L, fwhm3_G, fwhm3_L = theta
        a = Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G) + Voigt1D(x_2, amplitude2_L, fwhm2_L, fwhm2_G) + Voigt1D(x_3, amplitude3_L, fwhm3_L, fwhm3_G)
        return a(x)

    def model_Voigt(self, theta,x):
        x_0, amplitude_L, fwhm_G, fwhm_L = theta
        a = Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G)
        return a(x)

    def model_Gauss(self, theta,x):
        mean, amplitude, std = theta
        a = Gaussian1D(amplitude, mean, std)
        return a(x)

    def model_Lorentz(self, theta,x):
        mean, amplitude, fwhm = theta
        a = Lorentz1D(amplitude, mean, fwhm)
        return a(x)

    def lnlike(self, theta, x, y, yerr, model):
        
        """
        Here we compute the likelihood of the current model given the data.
        """
        
        return -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)
        
    def lnprior(self, theta, priors):
        
        """
        This function checks if the input parameters satisfy the prior conditions.
        """
        
        for i in range(len(theta)):
            if priors[i][0] < theta[i] < priors[i][1]:
                continue
            else:
                return -np.inf
        
        return 0.0
        
    def lnprob(self, theta, priors, x, y, yerr, model):
        lp = self.lnprior(theta, priors)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr, model)

    def line_MCMC(self, p0, priors, nwalkers, niter, initial, lnprob, data, model_name, custom_function=None, burn_in=100, ncores=1):
        
        priors = tuple([priors])  # The priors are transformed into a tuple containing only one element
        
        if model_name == "custom":
            if not callable(custom_function):
                raise Exception("Parameter custom_function is not callable. This parameter must be a function to work properly.")
                
        adopted_model = self.all_models(model_name, custom_function)  # Here we choose a function for the fit
        adopted_model = tuple([adopted_model])  # The adopted model is transformed into a tuple containing only one element
        metadata = priors + data + adopted_model
        
        if ncores > 1:
            if OS_name == "Darwin":
                warnings.warn("Multiprocessing may not work well in MacOS. If you have problems, try to do the single-core processing, i.e. ncores = 1.")

            with Pool(ncores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, len(initial), lnprob, args=metadata, pool=pool)
                start = time.time()
                print("Running burn-in...")
                p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=True)
                sampler.reset()
                
                print("Running production...")
                pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        else:
            sampler = emcee.EnsembleSampler(nwalkers, len(initial), lnprob, args=metadata)
            
            start = time.time()
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=True)
            sampler.reset()

            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
            end = time.time()
            serial_time = end - start
            print("Single-core processing took {0:.1f} seconds".format(serial_time))
        return sampler, pos, prob, state
    

    def plotter(self, sampler, model_name, x, color="grey", custom_function=None, normalization = 1):
    
        if model_name == "custom":
            if not callable(custom_function):
                raise Exception("Parameter custom_function is not callable. This parameter must be a function to work properly.")
        else:
            if custom_function is not None:
                custom_function = None
                
        samples = sampler.flatchain
        adopted_model = self.all_models(model_name, custom_function)
        
        for theta in samples[np.random.randint(len(samples), size=100)]:
            plt.plot(x, adopted_model(theta, x)*normalization, color=color, zorder=0, alpha=0.1)  # plotting with parameters in the posterior destribution

        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        plt.grid(linestyle=":")
        plt.legend()

    def MCMC_spread(self, x, samples, model_name, nsamples=100, custom_function=None):
        
        if model_name == "custom":
            if not callable(custom_function):
                raise Exception("Parameter custom_function is not callable. This parameter must be a function to work properly.")
        else:
            if custom_function is not None:
                custom_function = None
        
        models = []
        draw = np.floor(np.random.uniform(0,len(samples),size=nsamples)).astype(int)
        thetas = samples[draw]  # Each element of thetas contain the N parameters of the assumed model
        adopted_model = self.all_models(model_name, custom_function)
        for theta in thetas:
            mod = adopted_model(theta,x)
            models.append(mod)
        spread = np.std(models,axis=0)
        median_model = np.median(models,axis=0)
        return median_model,spread
        
    def data_window_selection(self, wavelengths, spec_flux, spec_flux_err, line_region_min,line_region_max):
        
        """
        wavelength must be in absolute values, i.e. without units
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

    def parameter_estimation(self, samples, model_name, air_wavelength_line=None, quantiles=[0.16, 0.5, 0.84], normalization = 1, parlabels=None, line_names="", output_dir=".", savefile=True):
        
        """
        line_names: str or list of str
        
        parlabels: list
        this is a list with the names of the free parameters in the adopted model. If you use a **custom** model
        with two peaks, then parlabels needs two input lists, e.g.: parlabels=[['a','b'],['c','d']]. If it has three peaks, parlabels
        must be parlabels=[['a','b'],['c','d'],['e','f']] and so on.
        
        lambda_peak_names: str or list of str
        Name or list of names of the parameters corresponting to the line peaks. E.g.: for a Gaussian, the parameter that tell us where
        the line peak is located is the Gaussian mean. E.g. 2: if your model consists of two Gaussians with 
        parlabels = [["mean", "Amplitude", "fwhm"],["mean2", "Amplitude2", "fwhm2"]], then
        lambda_peak_names = ["mean","mean2"]
        
        """
        
        if isinstance(parlabels[0],str):
            parlabels = [parlabels]

        if model_name == "custom":
            if parlabels is None:
                raise Exception("For a custom model, it is mandatory to input the list 'parlabels'.")
            
        output_dir = str(Path(output_dir))
        
        
        lambda_peak_names = ["Mean"]
        if model_name[0:6] == "double":
            lambda_peak_names.append("Mean2")
        elif model_name[0:6] == "triple":
            lambda_peak_names.append("Mean2")
            lambda_peak_names.append("Mean3")
        
         
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
        for i in range(len(lambda_peak_names)):
            if savefile:
                if model_name == "custom":
                    f = open(f'{output_dir}/{self.target_name}_{line_names[i]}_line_fit_results_custom_model.csv','w')
                else:
                    f = open(f'{output_dir}/{self.target_name}_{line_names[i]}_line_fit_results.csv','w')
                f.write("# Parameter name, value, error_down, error_up\n")
            par_values = []
            par_values_errors = []
            par_names = []
            ndim = len(parlabels[i])  # number of dimensions/parameters

            for j in range(ndim):  # must be done once per variable
                q_16, q_50, q_84 = corner.quantile(samples[:,j+previous_j], quantiles)
                dx_down, dx_up = q_50-q_16, q_84-q_50
                # Computing the redshift of the line:
                if parlabels[i][j] == lambda_peak_names[i] and air_wavelength_line is not None:
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
                    parlabels[i][j] = "fwhm_Gauss"+parlabels[i][j][3:]
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
                    if parlabels[i][j][0:9] == "Amplitude":
                        f.write(f"{parlabels[i][j]}, {par_values[-1]}, {par_values_errors[-1][0]}, {par_values_errors[-1][1]}\n")
                    else:
                        f.write(f"{parlabels[i][j]}, {par_values[-1]}, {par_values_errors[-1][0]}, {par_values_errors[-1][1]}\n")
            
            previous_j = j + previous_j + 1
            if savefile:
                f.close()
        
        return par_values, par_values_errors, par_names


    def quick_plot(self, x,y,model_name, custom_function=None, sampler=None, best_fit_model=None, theta_max=None, normalization= 1, hair_color="grey", title="",xlabel="Observed $\lambda$ [$\AA$]",ylabel="F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ ]",savefig=True, outputdir="./"):
        
        """
        theta_max: numpy.ndarray
        This parameter is useful here only if model_name is not 'custom'. I will be used to plot the vertical lines
        coincident with the line peaks.
        
        
        """
        
        if model_name == "custom":
            if not callable(custom_function):
                raise Exception("Parameter custom_function is not callable. This parameter must be a function to work properly.")
        else:
            if custom_function is not None:
                custom_function = None
                
        f = plt.figure(figsize=(10,8))
        ax = f.add_subplot(1,1,1)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='major', length=5, direction='in')
        ax.tick_params(which='minor', length=2.5, direction='in',bottom=True, top=True, left=True, right=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        
        plt.plot(x,y*normalization,color="orange", label="data after cont. subtraction")
        if sampler is not None:
            self.plotter(sampler,model_name,x,color=hair_color, custom_function=custom_function, normalization = normalization)
        if best_fit_model is not None:
            plt.plot(x,best_fit_model*normalization, color="black", label="Highest Likelihood Model")
        plt.xlabel(xlabel,fontsize=12)
        plt.ylabel(ylabel,fontsize=12)
        plt.title(title)
        plt.grid(which="both", linestyle=":")
        plt.ticklabel_format(scilimits=(-5, 8))
        plt.ylim((y.min()-0.1*y.max())*normalization,1.1*y.max()*normalization)
        
        if theta_max is not None:
            if model_name == "doubleVoigt":
                plt.plot(x,self.model_Voigt(theta_max[0:4],x)*normalization, color="k", linestyle=":", label="Voigt components") 
                plt.plot(x,self.model_Voigt(theta_max[4:],x)*normalization, color="k", linestyle=":")
                plt.vlines(theta_max[0],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
                plt.vlines(theta_max[4],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
            elif model_name == "tripleVoigt":
                plt.plot(x,self.model_Voigt(theta_max[0:4],x)*normalization, color="k", linestyle=":", label="Voigt components")
                plt.plot(x,self.model_Voigt(theta_max[4:8],x)*normalization, color="k", linestyle=":")
                plt.plot(x,self.model_Voigt(theta_max[8:],x)*normalization, color="k", linestyle=":")
                plt.vlines(theta_max[0],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
                plt.vlines(theta_max[4],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
                plt.vlines(theta_max[8],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
            elif model_name != "custom":
                plt.vlines(theta_max[0],y.min()-0.1*y.max(),10000, colors="k", linewidth=0.5)
        plt.legend()
        if savefig:
            plt.savefig(outputdir+"/"+title+"_line.png")
        return

    def merge_fit_results(self, target_name,list_of_files=None, wavelength_systematic_error = None, rest_frame_line_wavelengths = None, output_dir="./"):
        
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
        minimum_list_of_parameter_names = np.asarray(['z','Mean','Amplitude','fwhm_Lorentz','fwhm_Gauss'])
        if len(parameter_names) < len(minimum_list_of_parameter_names):
            parameter_names = minimum_list_of_parameter_names
        f.write("# Line name")
        for parameter_name in parameter_names:
            if parameter_name[0:4] == "Mean":
                f.write(", "+parameter_name+" [Ang], "+parameter_name+"_error_down, "+parameter_name+"_error_up")
            elif parameter_name[0:4] == "Ampl":
                f.write(", "+parameter_name+" (flux_dens - continuum) [erg cm-2 s-1 Ang-1], "+parameter_name+"_error_down, "+parameter_name+"_error_up")
            elif parameter_name[0:4] == "fwhm":
                f.write(", "+parameter_name+" [Ang], "+parameter_name+"_error_down, "+parameter_name+"_error_up")
            else:
                f.write(", "+parameter_name+", "+parameter_name+"_error_down, "+parameter_name+"_error_up")
        
        if wavelength_systematic_error is not None:
            f.write(", Systematic wavelength error [Ang], Systematic z error")

        # Writing down the parameters and filling the empty spaces with zeros:
        for i in range(len(data_list)):

            f.write("\n"+line_names[i])
            line_array = np.asarray(data_list[i])

            for number,parameter in enumerate(line_array):
                if parameter[0] in parameter_names:
                    index = np.where(parameter_names == parameter[0])[0][0]
                    if index == number:
                        f.write(", "+parameter[1]+", "+str(np.abs(float(parameter[2])))+", "+str(np.abs(float(parameter[3]))))
                    else:
                        f.write(", 0, 0, 0, "+parameter[1]+", "+str(np.abs(float(parameter[2])))+", "+str(np.abs(float(parameter[3]))))
                    
                if parameter[0][0:12] == "fwhm_Lorentz":
                    try:
                        if line_array[number+1][0][0:10] != "fwhm_Gauss": # Here it could be different of anything. It is just a generic condition.
                            f.write(", 0, 0, 0")
                    except:
                        f.write(", 0, 0, 0")

            if wavelength_systematic_error is not None:
                f.write(f", {wavelength_systematic_error}, {wavelength_systematic_error/rest_frame_line_wavelengths[i]}")

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
        
        """
        
        if which_model == "Gaussian" or which_model == "gaussian":
            initial = np.array([observed_wavelength, peak_height, 10])
            priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150]])
            labels = ["Mean", "Amplitude", "std"]
            adopted_model = self.model_Gauss
        elif which_model == "Lorentz" or which_model == "lorentz":
            initial = np.array([observed_wavelength, peak_height, 10])
            priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150]])
            labels = ["Mean", "Amplitude", "fwhm_Lorentz"]
            adopted_model = self.model_Lorentz
        elif which_model == "Voigt" or which_model == "voigt":
            initial = np.array([observed_wavelength, peak_height, 10, 10])
            priors = np.array([[line_region_min,line_region_max],[0.1*peak_height, 10*peak_height],[0.1,150],[0.1,150]])
            labels = ["Mean", "Amplitude", "fwhm_Lorentz", "fwhm_Gauss"]
            adopted_model = self.model_Voigt

        """if which_model == "doubleVoigt":
            labels = ["Mean", "Amplitude", "fwhm_L", "fwhm_G", "Mean2", "Amplitude2", "fwhm2_L", "fwhm2_G"]
        elif which_model == "tripleVoigt":
            labels = ["Mean", "Amplitude", "fwhm_L", "fwhm_G", "Mean2", "Amplitude2", "fwhm2_L", "fwhm2_G", "Mean3", "Amplitude3", "fwhm3_L", "fwhm3_G"]"""

        return initial, priors, labels, adopted_model



    def fit_lines(self, wavelengths, flux_density, continuum_baseline, wavelength_peak_positions, rest_frame_line_wavelengths, peak_heights, line_std_deviation,
                  which_models="Lorentz", line_names = None, overplot_archival_lines = ["H"], priors = None, MCMC_walkers = 250,
                  MCMC_iterations = 400, N_cores = 1, plot_spec = True, plot_MCMC = False, save_results = True):

        """
        This function uses a Markov-chain Monte Carlo to estimate the line parameters and their errors.

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
        peak_heights: numpy.ndarray (astropy.units Angstrom)
            The height of each peak in erg/cm2/s/A. This variable is an output of the function analysis.find_lines().
        line_std_deviation: numpy.ndarray (float)
            The standard deviation for the local continuum. This variable is an output of the function analysis.find_lines().
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
            This is the number of walkers for the MCMC.
        MCMC_iterations: int
            This is the number of iterations for the MCMC.
        N_cores: int
            This is the number of cores in case you want to run this analysis in parallel.
        plot_spec: boolean
            If True, a plot of the spectrum with the lines requested in the input variable overplot_archival_lines will be shown.
        plot_MCMC: boolean
            If True, a series of diagnostic plots for the MCMC will be shown, as the corner plot, the evolution of the parameters over time, and the line fitted to 
            the data.
        save_results: boolean
            If True, the plots and fit information will be saved in the output directory defined in the function analysis.load_calibrated_data()
            
        Returns
        -------
        line_names: list
            A list with the names of the lines. This is useful only if the input variable line_names is None.
        par_values_list: list
            This is a list containing sublists with the best-fit values for each line model.
        par_values_errors_list: list
            A list with the asymmetrical errors for each parameter listed in par_values_list.
        par_names_list: list
            A list with the names of all the parameters used in each line model.
        XXXXXXX_lines.csv:
            Optional. This file contains all the best-fit parameters for each line model and is saved in the output directory defined in the
            function analysis.load_calibrated_data(). "XXXXXXX" here stands for the target's name.
        """

        if isinstance(wavelength_peak_positions, u.quantity.Quantity):
            wavelength_peak_positions = wavelength_peak_positions.value
        
        if isinstance(wavelength_peak_positions,float) or isinstance(wavelength_peak_positions,int):
            wavelength_peak_positions = [wavelength_peak_positions]

        if isinstance(peak_heights, u.quantity.Quantity):
            peak_heights = peak_heights.value

        if isinstance(peak_heights,float) or isinstance(peak_heights,int):
            peak_heights = np.asarray([peak_heights])
        elif isinstance(peak_heights,list):
            peak_heights = np.asarray(peak_heights)
        
        if isinstance(line_std_deviation,float) or isinstance(line_std_deviation,int):
            line_std_deviation = np.asarray([line_std_deviation])
        elif isinstance(line_std_deviation,list):
            line_std_deviation = np.asarray(line_std_deviation)

        if isinstance(rest_frame_line_wavelengths,float) or isinstance(rest_frame_line_wavelengths,int):
            rest_frame_line_wavelengths = [rest_frame_line_wavelengths]

        if isinstance(which_models,str):
            which_models = [which_models]*len(rest_frame_line_wavelengths)
        elif isinstance(which_models,list):
            if len(which_models) != len(rest_frame_line_wavelengths) or len(which_models) != len(wavelength_peak_positions):
                raise RuntimeError("Input variables 'which_models', 'rest_frame_line_wavelengths', and 'wavelength_peak_positions' must have the same size.")

        if line_names is None:
            line_names = []
            for number, _ in enumerate(rest_frame_line_wavelengths): 
                line_names.append(f"line_{number}")


        local_continuum_index = []
        for wavelength in wavelength_peak_positions:
            local_continuum_index.append(extraction.find_nearest(wavelengths.value, wavelength))
        peak_heights = peak_heights - continuum_baseline[local_continuum_index]
        normalization = 10**round(np.log10(np.median(flux_density.value - 0.9*continuum_baseline)))

        par_values_list, par_values_errors_list, par_names_list = [], [], []
        for number, peak_height in enumerate(peak_heights):
            invert_spectrum = 1  # For emission lines, the spectrum is multiplied by 1. For absorption lines, it is multiplied by -1
            if peak_height < 0:
                invert_spectrum = -1
            local_normalization = invert_spectrum*normalization
            continuum_subtracted_flux = (flux_density.value - continuum_baseline)/local_normalization
            peak_height = peak_height/local_normalization
            local_line_std_deviation = line_std_deviation[number]/local_normalization

            if priors is None or priors[number] is None:
                line_region_min = wavelength_peak_positions[number] - 100
                if number > 0:
                    mean_point = (wavelength_peak_positions[number-1] + wavelength_peak_positions[number])/2
                    if mean_point > line_region_min:
                        line_region_min = mean_point

                line_region_max = wavelength_peak_positions[number] + 100
                if number < (len(rest_frame_line_wavelengths)-1):
                    mean_point = (wavelength_peak_positions[number] + wavelength_peak_positions[number+1])/2
                    if mean_point < line_region_max:
                        line_region_max = mean_point

                initial, local_priors, labels, adopted_model = self.automatic_priors(which_models[number], wavelength_peak_positions[number], peak_height, line_region_min, line_region_max)
            else:
                initial, _, labels, adopted_model = self.automatic_priors(which_models[number], wavelength_peak_positions[number], peak_height, line_region_min=None, line_region_max=None)
                local_priors = np.asarray(priors[number],dtype="object")
                # In the case of a single-line analysis, if the user inputs priors=[[7500,7700],[0.1],[2,50]] instead of priors=[ [[7500,7700],[0.1],[2,50]] ], the analysis will work anyway.
                if isinstance(local_priors[0],float) or isinstance(local_priors[0],int):
                    local_priors = np.asarray(priors,dtype="object")
                line_region_min = local_priors[0][0]
                line_region_max = local_priors[0][1]
            

            
            x,y,yerr = self.data_window_selection(wavelengths.value, continuum_subtracted_flux, local_line_std_deviation*np.ones(len(wavelengths.value)), line_region_min, line_region_max)
            data = (x, y, yerr)
            p0 = [np.array(initial) + 0.01 * np.random.randn(len(initial)) for i in range(MCMC_walkers)]  # p0 is the methodology of stepping from one place on a grid to the next.
            sampler, pos, prob, state = self.line_MCMC(p0, local_priors, MCMC_walkers, MCMC_iterations, initial, self.lnprob, data, which_models[number], ncores=N_cores)
            samples = sampler.flatchain
            theta_max = samples[np.argmax(sampler.flatlnprobability)]

            par_values, par_values_errors, par_names = self.parameter_estimation(samples, which_models[number], air_wavelength_line = rest_frame_line_wavelengths[number],
                                                                                 normalization=local_normalization, parlabels = list.copy(labels), line_names=line_names[number],
                                                                                 output_dir = self.output_dir, savefile=save_results)
            par_values_list.append(par_values)
            par_values_errors_list.append(par_values_errors)
            par_names_list.append(par_names)

            best_fit_model = adopted_model(theta_max, x)

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

                self.quick_plot(x,y,model_name=which_models[number],sampler=sampler,best_fit_model=best_fit_model,theta_max=theta_max, normalization=local_normalization,
                                hair_color="grey",title=self.target_name+" - "+line_names[number], ylabel="F$_{\lambda}$ ["+f"{flux_density.unit}]", outputdir = self.output_dir)

        redshifts = np.asarray(par_values_list, dtype=object) # The dtype option here is usefull because our par_values can have different sizes
        average_redshift = np.average(redshifts[:,0])
        std_redshift = np.std(redshifts[:,0])        
        
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
            self.merge_fit_results(self.target_name,list_of_files=None, wavelength_systematic_error=self.wavelength_systematic_error.value,
                                   rest_frame_line_wavelengths = rest_frame_line_wavelengths,output_dir=self.output_dir)
            list_of_files = glob.glob(self.output_dir+"/*_line_fit_results.csv")
            for file in list_of_files:
                os.remove(file)
        
        if plot_spec or plot_MCMC:
            plt.show()

        return line_names, par_values_list, par_values_errors_list, par_names_list
    

"""
If we were dealing with an absorption line, we could compute its equivalent width, which is defined to be the width of a rectangle that has the same
integral as the absorption line, but goes all the way from the continuum level to zero: https://en.wikipedia.org/wiki/Equivalent_width

We could compute this with our model, assuming our continuum is flat (has zero slope):

EQW = -absorption_fit(wavelengths.value[selection]).sum() / continuum_fit.intercept * u.nm

Function to compute velocity dispersion, integrated flux for each line, and EQW.

Double and triple lines.
"""