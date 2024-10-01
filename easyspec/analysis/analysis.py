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
from easyspec.cleaning import cleaning
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter
from astropy import units as u
from astropy.modeling.models import Linear1D
from scipy.signal import medfilt
import astroquery

plt.rcParams.update({'font.size': 12})


easyspec_analysis_version = "1.0.0"


class analysis:

    """This class contains all the functions necessary to perform the analysis of calibrated spectral data."""

    def __init__(self): 
        # Print the current version of easyspec-extraction       
        print("easyspec-analysis version: ",easyspec_analysis_version)
    
    def continuum_fit(self,flux, wavelengths, continuum_regions, method = "powerlaw", pl_order=2, smooth_window=111):
    
        """
        This function fits the continuum emission with an spline or a power law. For a spline, it can also smooth the continuum if requested.

        Parameters
        ----------
        flux: array (float)
            Array with the flux to be fitted and (if spline is chosen) smoothed. 
        wavelengths: array (float)
            Wavelength must be in absolute values, i.e. without units.
        continuum_regions: list of intervals (float)
            The continuum will be computed only within these wavelength regions and then extrapolated everywhere else. E.g.: continuum_regions = [[3000,6830],[6930,7500]].
            If None, then all the wavelengths will be used.
        pl_order: int
            Polynomial order for the power law fit.

        Returns
        -------
        continuum_selection: array (float)
            Array with the continuum flux.
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
        
    def load_calibrated_data(self, calibrated_spec_data, target_name = None, plot = True):

        """
        
        calibrated_spec_data: pode ser o endereco, uma lista de enderecos, ou os dados obtidos em extraction.

        Parameters
        ----------
        target_spec_file: string 
            The path to the data '.fits' or '.fit' file containing the target spectrum.


        Returns
        -------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        """

        if target_name is None:
            target_name = calibrated_spec_data.split(".")[-1]
        self.target_name = target_name

        data = np.loadtxt(calibrated_spec_data)
        wavelengths, flux = data[:,0]*u.angstrom, data[:,1]*u.erg / u.cm**2 / u.s / u.AA  # Angstrom, erg/cm2/s/Angstrom

        if plot:
            plt.figure(figsize=(12,5))
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(0,flux.value.max()*1.2)
            plt.title(f"Calibrated data -  {target_name}")
            plt.plot(wavelengths, flux, color='orange', label=target_name+" - calibrated data")
            plt.legend()
            plt.ylabel("F$_{\lambda}$ "+f"[{flux.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.show()

        return wavelengths, flux

    def find_lines(self, wavelengths, flux, line_type="emission", line_significance_sigma = 5, peak_distance = 30, peak_width = 10, method = "median_filter", continuum_regions = None, pl_order=2, smooth_window=111, plot_lines = True, plot_regions = True):

        """
        Use a line library like https://astronomy.nmsu.edu/drewski/tableofemissionlines.html, ou MELHOR: astroquery.nist

        Estimar um continuum_std_deviation dinamico, cioe, que depende do comprimento de onda. Provavelmente o melhor a se fazer eh tornar
        obrigatorio o continuum_regions e calcular os sigmas a partir dele.
        

        Parameters
        ----------
        line_significance_sigma: float
            Defines how many standard deviations above the continuum the line peak must be in order to be detected.

        Returns
        -------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.

        line_significance: array (floats)
            The line significance with respect to the local continuum standard deviation.
        """

        if isinstance(continuum_regions[0],float) or isinstance(continuum_regions[0],int):
            continuum_regions = [continuum_regions]

        continuum_baseline, continuum_std_deviation = self.continuum_fit(flux.value, wavelengths.value, method = method, continuum_regions = continuum_regions, pl_order = pl_order, smooth_window = smooth_window) 
        
        peak_heights = np.asarray([])
        peak_position_index = np.asarray([])
        line_significance = np.asarray([])
        if plot_regions:
            plt.figure(figsize=(12,4))
            plt.ylabel("F$_{\lambda}$ "+f"[{flux.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.title("The continuum noise is independently estimated for each one of these regions")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())

        # Below we do a loop over the values of standard deviation for the selected continuum regions. The line significance is estimated based on the standard deviation of the closest continuum region.
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
            continuum_removed_flux = flux.value[index]-continuum_baseline[index]
            if plot_regions:
                plt.plot(wavelengths.value[index], continuum_removed_flux) 
            if line_type == "emission":
                local_peak_position_index, local_peak_heights = scipy.signal.find_peaks(continuum_removed_flux,height=peak_height,distance = peak_distance, width = peak_width)
                peak_heights = np.concatenate([peak_heights,local_peak_heights["peak_heights"]])
                peak_position_index = np.concatenate([peak_position_index, local_peak_position_index + index.min()])
                line_significance = np.concatenate([line_significance,local_peak_heights["peak_heights"]/std_deviation])
            elif line_type == "absorption":
                local_peak_position_index, local_peak_heights = scipy.signal.find_peaks(-1*continuum_removed_flux,height=peak_height,distance = peak_distance, width = peak_width)
                peak_heights = np.concatenate([peak_heights,-1*local_peak_heights["peak_heights"]])
                peak_position_index = np.concatenate([peak_position_index, local_peak_position_index + index.min()])
                line_significance = np.concatenate([line_significance,local_peak_heights["peak_heights"]/std_deviation])
                ylim_min = peak_heights.max()
            else:
                raise RuntimeError("The input options for the line_type variable are 'emission' or 'absorption'.")
        

        peak_position_index = peak_position_index.astype(int)
        peak_heights = (peak_heights+continuum_baseline[peak_position_index])*flux.unit
        wavelength_peak_positions = wavelengths[peak_position_index]
        continuum_removed_flux = flux.value-continuum_baseline

        if plot_lines:
            plt.figure(figsize=(12,5))
            if len(peak_position_index) > 0:
                if line_type == "emission":
                    for number, peak_wavelength in enumerate(wavelength_peak_positions.value):
                        plt.text(peak_wavelength-35, peak_heights.value[number] + 0.05*peak_heights.value.max(), str(round(peak_wavelength,3))+"$\AA$",rotation=90,fontsize=10)
                else:
                    for number, peak_wavelength in enumerate(wavelength_peak_positions.value):
                        plt.text(peak_wavelength-35, peak_heights.value[number] - 0.05*peak_heights.value.min(),str(round(peak_wavelength,3))+"$\AA$",rotation=90,fontsize=10, horizontalalignment="left")
            
            if method != "median_filter":
                plt.plot(wavelengths,continuum_baseline,label="Power-law continuum")
            else:
                plt.plot(wavelengths,continuum_baseline,label="Median-filter continuum")

            plt.plot(wavelengths, flux, color='orange')
            plt.plot(wavelengths, continuum_removed_flux, alpha=0.15, color='black', label="Continuum-subtracted spec")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            if line_type == "emission":
                plt.ylim(0,flux.value.max()*1.4)
            else:
                plt.ylim(ylim_min,flux.value.max()*1.2)
            plt.ylabel("F$_{\lambda}$ "+f"[{flux.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.title(self.target_name)
            plt.legend()
        
        if plot_lines or plot_regions:
            plt.show()
        
        return continuum_baseline, continuum_std_deviation, wavelength_peak_positions, peak_heights, line_significance