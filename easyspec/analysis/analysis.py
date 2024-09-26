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

plt.rcParams.update({'font.size': 12})


easyspec_analysis_version = "1.0.0"


class analysis:

    """This class contains all the functions necessary to perform the analysis of calibrated spectral data."""

    def __init__(self): 
        # Print the current version of easyspec-extraction       
        print("easyspec-analysis version: ",easyspec_analysis_version)
    
    def continuum_fit(self,flux, wavelengths, continuum_regions, pl_order=2):
    
        """
        This function fits the continuum emission with an spline or a power law. For a spline, it can also smooth the continuum if requested.

        Parameters
        ----------
        flux: array (float)
            Array with the flux to be fitted and (if spline is chosen) smoothed. 
        wavelengths: array (float)
            Wavelength must be in absolute values, i.e. without units.
        continuum_regions: list of intervals (float)
            The continuum will be computed only within these wavelength regions and then extrapolated everywhere else. E.g.: continuum_regions = [[3000,6830],[6930,7500]]
        pl_order: int
            Polynomial order for the power law fit.

        Returns
        -------
        continuum_selection: array (float)
            Array with the continuum flux.
        continuum_std_deviation: float
            The standard deviation for the continuum.
        """
        
        selection = np.asarray([])
        for wavelength_region in continuum_regions:
            # Here we select the indexes of the wavelengths inside the intervals given by continuum_regions:
            index = np.where((wavelengths > wavelength_region[0]) & (wavelengths < wavelength_region[1]))[0]
            selection = np.concatenate([selection,index])
        
        selection = selection.astype(int)
        flux_continuum = flux[selection]
        wavelengths_continuum = wavelengths[selection]

        z = np.polyfit(wavelengths_continuum, flux_continuum, pl_order)
        continuum_selection = np.poly1d(z)
        continuum_selection = continuum_selection(wavelengths)
        continuum_std_deviation = np.std(flux_continuum-continuum_selection[selection])

        return continuum_selection, continuum_std_deviation
        
    def load_calibrated_data(self, calibrated_spec_data, target_name, plot = True):

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

        data = np.loadtxt(calibrated_spec_data)
        wavelengths, flux = data[:,0]*u.angstrom, data[:,1]*u.erg / u.cm**2 / u.s / u.AA  # Angstrom, erg/cm2/s/Angstrom

        if plot:
            plt.figure(figsize=(12,5))
            plt.plot(wavelengths, flux, color='orange')
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(0,flux.value.max()*1.2)
            plt.title(f"Calibrated spec {target_name}")
            plt.ylabel("F$_{\lambda}$ "+f"[{flux.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.legend()
            plt.show()

        return wavelengths, flux

    def find_lines(self, wavelengths, flux, continuum_regions, line_type="emission", line_significance = 5, peak_distance = 30, peak_width = 10, pl_order=2, plot = True):

        """

        DEVELOP THE line_type="absorption" CASE!!!

        Parameters
        ----------
        line_significance: float
            Defines how many standard deviations above the continuum the line peak must be in order to be detected.

        Returns
        -------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        """

        continuum_baseline, continuum_std_deviation = self.continuum_fit(flux.value, wavelengths.value, continuum_regions, pl_order=pl_order) 
        print("Continuum std deviation: ",continuum_std_deviation)
        peak_height = line_significance*continuum_std_deviation
        continuum_removed_flux = flux.value-continuum_baseline
        peak_position_index, peak_heights = scipy.signal.find_peaks(continuum_removed_flux,height=peak_height,distance = peak_distance, width = peak_width)
        peak_heights = peak_heights["peak_heights"]
        print("peak heights: ", peak_heights)
        print("peak positions: ",wavelengths[peak_position_index])

        if plot:
            plt.figure(figsize=(12,5))
            if len(peak_position_index) > 0:
                for number, peak_wavelength in enumerate(wavelengths[peak_position_index].value):
                    plt.text(peak_wavelength-25, peak_heights[number]+continuum_baseline[peak_position_index][number] + 3*continuum_std_deviation,str(round(peak_wavelength,3))+"$\AA$",rotation=90,fontsize=10)
            
            plt.plot(wavelengths,continuum_baseline,label="PL continuum")

            plt.plot(wavelengths, flux, color='orange')
            plt.plot(wavelengths, continuum_removed_flux, alpha=0.15, color='black', label="Continuum-subtracted spec")
            plt.minorticks_on()
            plt.grid(which="both",linestyle=":")
            plt.xlim(wavelengths.value.min(),wavelengths.value.max())
            plt.ylim(0,flux.value.max()*1.3)
            plt.ylabel("F$_{\lambda}$ "+f"[{flux.unit}]",fontsize=12)
            plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
            plt.legend()
            plt.show()
        
        return continuum_baseline