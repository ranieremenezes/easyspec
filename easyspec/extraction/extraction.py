from astropy.io import fits
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import factorial
import scipy
from astropy.modeling.models import Gaussian1D, Voigt1D, Lorentz1D
from astropy.modeling.fitting import LevMarLSQFitter
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
from astropy.modeling.fitting import LinearLSQFitter

cleaning = cleaning()


plt.rcParams.update({'font.size': 12})

easyspec_extraction_version = "1.0.0"


class extraction:

    """This class contains all the functions necessary to perform the spectral extraction from a cleaned/stacked image."""

    def __init__(self): 
        # Print the current version of easyspec-extraction       
        print("easyspec-extraction version: ",easyspec_extraction_version)

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
        if len(value) > 1:
            idx = []
            for i in value:
                idx.append((np.abs(array - i)).argmin())      
            return idx
        else: 
            idx = (np.abs(array - value)).argmin()
            return idx

    def savitzky_golay(self,y, window_size, order, deriv=0, rate=1):
    
        r"""
        This function was taken from https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay

        Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.

        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)

        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal.

        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over an odd-sized window centered at
        the point.

        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """
        
        try:
            window_size = np.abs(int(window_size))
            order = np.abs(int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))

        return np.convolve( m[::-1], y, mode='valid')
    
    def continuum_fit(self,diff_flux, wavelengths, continuum_regions, bkg_model="spline", pl_order=2, window_size=11, smoothing_order=3, smoothing=True, interpolation_order=2):
    
        """
        This function fits the continuum emission with an spline or a power law. For a spline, it can also smooth the continuum if requested.

        Parameters
        ----------
        diff_flux: array (float)
            Array with the differential flux to be fitted and (if spline is chosen) smoothed. 
        wavelengths: array (float)
            Wavelength must be in absolute values, i.e. without units.
        continuum_regions: list of intervals (float)
            The continuum will be computed only within these wavelength regions and then extrapolated everywhere else. E.g.: continuum_regions = [[3000,6830],[6930,7500]]
        bkg_model: string
            Model for extracting the continuum. Options are "spline" or "pl" (power law). 
        pl_order: int
            Polynomial order for the power law fit.
        window_size: int
            Window size for the smoothing Savitzky-Golay algorithm. Must be an **odd** integer number.
        smoothing_order: int
            Polynomial order for the smoothing. Must be less then `window_size` - 1.
        smoothing: bool
            Set to True if you want to smooth the continuum spectrum.
        interpolation_order: int
            The interpolation order when extracting the continuum.

        Returns
        -------
        continuum_selection: array (float)
            Array with the continuum differential flux.
        """
        
        selection = np.asarray([])
        for wavelength_region in continuum_regions:
            # Here we select the indexes of the wavelengths inside the intervals given by continuum_regions:
            index = np.where((wavelengths > wavelength_region[0]) & (wavelengths < wavelength_region[1]))[0]
            selection = np.concatenate([selection,index])
        
        selection = selection.astype(int)
        diff_flux_continuum = diff_flux[selection]
        wavelengths_continuum = wavelengths[selection]
        
        if bkg_model == "spline":
            tck_selection = interpolate.splrep(wavelengths_continuum, diff_flux_continuum, k=interpolation_order)
            continuum_selection = interpolate.splev(wavelengths, tck_selection)
            if smoothing:
                continuum_selection = self.savitzky_golay(continuum_selection, window_size, smoothing_order, deriv=0, rate=1)

        elif bkg_model == "PL" or bkg_model == "pl" or bkg_model == "power law" or bkg_model == "power-law" or bkg_model == "powerlaw":
            z = np.polyfit(wavelengths_continuum, diff_flux_continuum, pl_order)
            continuum_selection = np.poly1d(z)
            continuum_selection = continuum_selection(wavelengths)

        return continuum_selection
    
    def import_data(self, target_spec_file, target_name, exposure_target = None, exposure_header_entry="AVEXP", reddening = None, std_star_spec_file = None, exposure_std_star = None, plot = True):

        """
        This function opens the data file (i.e. fits image with one or more spectra) and reads the exposure time from its header. Optionally, the user can manually add the exposure time.
        If you want to calibrate your spectrum in flux, please also add the standard star spectral fits file and the reddening.

        Parameters
        ----------
        target_spec_file: string 
            The path to the data '.fits' or '.fit' file containing the target spectrum.
        target_name: string
            The name of the target to be used in subsequent plots.
        exposure_target: float
            Optional. You can manually pass the exposure time (in seconds) of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'exposure_header_entry'
        exposure_header_entry: string
            Optional. If 'exposure_target' is not None, easyspec will automatically look in the image header for the exposure time corresponding to this entry.
        reddening: float
            Optional. The Galactic redening in the direction of your observation. You can look for the appropriate reddening here: 'https://irsa.ipac.caltech.edu/applications/DUST/' 
        std_star_spec_file: string
            Optional. The path to the data '.fits' or '.fit' file containing the standard star spectrum.
        exposure_std_star: float
            Optional. You can manually pass the exposure time (in seconds) of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'exposure_header_entry'
        plot: boolean
            If True, easyspec will plot the input spectra.


        Returns
        -------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        std_star_spec_data: numpy.ndarray (float)
            Matrix containing the standard star spectral image (only if the standard star spectral file is given by the user).
        """

        self.target_spec_data = fits.open(target_spec_file)[0].data
        self.target_name = target_name
        if exposure_target is None:
            self.exposure_target = cleaning.look_in_header(target_spec_file,exposure_header_entry)
        else:
            self.exposure_target = exposure_target
        if reddening is not None:
            self.reddening = reddening
        if std_star_spec_file is not None:
            self.std_star_spec_data = fits.open(std_star_spec_file)[0].data
        if exposure_std_star is not None:
            self.exposure_std_star = exposure_std_star
        elif std_star_spec_file is not None:
            self.exposure_std_star = cleaning.look_in_header(std_star_spec_file,exposure_header_entry)

        if plot:
            cleaning.plot(image_data=self.target_spec_data,figure_name=target_name,save=False)
            if std_star_spec_file is not None:
                cleaning.plot(image_data=self.std_star_spec_data,figure_name="Standard star",save=False)

        if std_star_spec_file is not None:
            print("Returning both target and std star images...")
            return self.target_spec_data, self.std_star_spec_data
        else:
            print("No standard star file was found. Returning only the target image.")
            return self.target_spec_data
        
    def tracing(self, target_spec_data, method = "argmax", y_pixel_range = 15, xlims = None, poly_order = 2, trace_half_width = 7, prominence = 50, distance = 20, Number_of_slices = 20, peak_dispersion_limit = 3, main_plot = True, plot_residuals = True):

        """
        This function detects all the spectra available in an image and fits a polynomial to recover the trace of each one of them. 
        It can be used to recover the spectral spine (or trace) of one or several spectra in the same image.

        Parameters
        ----------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        method: string
            The method to be adopted in finding the trace. Options are "argmax" or "moments" if you want to extract only the strongest spectrum in the image, or "multi" if you
            want to extract more than one spectrum. The option "argmax" works very well although is limited by the pixel size. The option "moments" is suited only for spectra 
            with a very high signal-to-noise ratio. The option "multi" will use the "argmax" method to look for several spectra in a single image.
        y_pixel_range: integer
            Y-axis range of pixels to be used to extract a specific spectrum. The total y-axis interval adopted is 2*y_pixel_range.
        xlims: list or array
            A list/array with the two limits in the x-axis delimiting the region where we will look for the spectra. E.g. if your image x-axis has 2000 pixels but your spectrum
            goes from pixels ~700 up to pixel ~1800, you can set xlims = [700,1800] to avoid fitting regions of pure noise. This variable is useful only for the "argmax" or
            "moments" methods.
        poly_order: integer
            The order of the polynomial(s) used to fit the trace.
        trace_half_width: integer
            The half-width of the trace. Here it is useful only for plotting the data.
        prominence: float
            The vertical distance between the peak and its lowest contour line. The smaller this value, the more spectra you will find in your image.
        distance: float
            Required minimal distance (>= 1) in pixels between neighbouring peaks in the y-axis. Default is 20 pixels. If you set this value too high with respect to the y-axis
            resolultion of your image, this function will return undesired results.
        Number_of_slices: integer
            Useful only for the "multi" mode. This is the number of slices in which every spectra in the image is divided in the x-axis. It is used to set the best x-axis interval
            at which easyspec will fit the trace.
        peak_dispersion_limit: float
            Useful only for the "multi" mode. This parameter sets the dispersion limit (with respect to the trace slice of minimum dispersion) of a trace slice such that we can
            use this slice to fit the trace, i.e. if the trace dispersion in a slice is larger than peak_dispersion_limit*minimum_dispersion, this slice will be ignored in the
            fit, where the term 'minimum_dispersion' stands for the dispersion in the slice with minimum trace dispersion.
        main_plot: boolean
            If True, easyspec will plot all the traces over all the spectra (or the main spectra) in the image.
        plot_residuals: boolean
            If True, easyspec will plot the residuals for each trace fitted to the data.
        
        Returns
        -------
        fitted_polymodel: list
            List with the fitted spine for all spectra in the image or only the brighter spectrum.
        """

        polymodel = Polynomial1D(degree = poly_order)
        linfitter = LinearLSQFitter()

        yvals = []
        yvals.append( np.argmax(target_spec_data, axis=0) ) # Generates an array containing only the pixels with the maximum number of counts for each column
        median_trace_value = np.median(yvals[0])
        xvals = np.arange(np.shape(target_spec_data)[1])
        ymin = []
        ymin.append(int(median_trace_value - y_pixel_range))
        ymax = []
        ymax.append(int(median_trace_value + y_pixel_range))

        if xlims is None:
            xmin, xmax = 0, np.shape(target_spec_data)[1]
        else:
            xmin, xmax = xlims[0], xlims[1]
        
        fitted_polymodel = []
        bad_pixels = []
        if method == "argmax":
            bad_pixels.append( (yvals[0] < ymin[0]) | (yvals[0] > ymax[0]) | (xvals < xmin) | (xvals > xmax) )  # That "|" is a bitwise "or" with assignment
            fitted_polymodel.append(linfitter(polymodel, xvals[~bad_pixels[-1]], yvals[0][~bad_pixels[-1]]))
        
        elif method == "moments":
            yvals = []
            yaxis = np.repeat(np.arange(ymin[0], ymax[0])[:,None], target_spec_data[:,:].shape[1], axis=1)
            background = [] 
            for i in range(np.shape(target_spec_data[ymin[0]:ymax[0],:])[1]):
                background.append(np.median(target_spec_data[ymin[0]:ymax[0],i]))

            background = np.asarray(background*np.shape(target_spec_data)[0])
            background = background.reshape(np.shape(target_spec_data))  # Each *column* of this matrix is filled with the same background value
                
            # moment 1 is the data-weighted average of the Y-axis coordinates
            weight = target_spec_data[ymin[0]:ymax[0],:] - background[ymin[0]:ymax[0],:]
            negative_vals = weight < 0.0   #The signal must be zero in the signal-free region for m1 to return an accurate estimate of the location of the peak
            weight[negative_vals] = 0.0 # Padding the background negative values with zeros
            yvals.append( np.average(yaxis, axis=0, weights=weight) )
            
            bad_pixels.append( (yvals[0] > ymax[0]) | (yvals[0] < ymin[0]) | (xvals < xmin) | (xvals > xmax) )
            fitted_polymodel.append(linfitter(polymodel, xvals[~bad_pixels[-1]], yvals[0][~bad_pixels[-1]]))
        
        elif method == "multi":
            yvals = []
            multi_peaks = []
            for i in range(len(target_spec_data[0,:])):
                local_peaks = scipy.signal.find_peaks(target_spec_data[:,i],prominence=prominence,width=None,threshold=None,distance=distance,height=None)[0]
                multi_peaks.append(local_peaks)
            
            multi_peaks = np.asarray(multi_peaks,dtype=object)
            flat_multi_peaks = multi_peaks[0]
            for i in multi_peaks[1:]:
                flat_multi_peaks = np.concatenate([flat_multi_peaks,i])
            
            peaks_histogram = np.histogram(flat_multi_peaks, bins=np.shape(target_spec_data)[0])
            final_peak_positions = scipy.signal.find_peaks(peaks_histogram[0],prominence=prominence,width=None,threshold=None,distance=distance,height=None)[0]
            print("Total number of spectra: ", len(final_peak_positions),", centered at y-pixels ", final_peak_positions)

            fitted_polymodel = []
            ymin = []
            ymax = []
            # The purpose of this loop is to find the minimum and maximum values in the x-axis at which it is still worth to use the peak
            for peak in final_peak_positions:
                ymin.append(int(peak - y_pixel_range))
                ymax.append(int(peak + y_pixel_range))
                yvals.append( np.argmax(target_spec_data[ymin[-1]:ymax[-1]], axis=0) + ymin[-1] )
                
                peak_dispersion = []
                spec_slices = np.linspace(0,target_spec_data.shape[1],Number_of_slices)
                interval = int(np.diff(spec_slices)[0])
                x_cut = []
                for i in spec_slices[:-1]:
                    i = int(i)
                    slice_dispersion = np.std(yvals[-1][i:i+interval])
                    if slice_dispersion < 1:
                        slice_dispersion = 1
                    peak_dispersion.append(slice_dispersion)
                    x_cut.append(xvals[i+interval])
                
                mininum_dispersion = np.asarray(peak_dispersion).min()
                for n,dispersion in enumerate(peak_dispersion):
                    if dispersion > peak_dispersion_limit*mininum_dispersion:
                        xmin = x_cut[n]
                    else:
                        break
                
                for n,dispersion in enumerate(np.flip(peak_dispersion)):
                    if dispersion > peak_dispersion_limit*mininum_dispersion:
                        xmax = x_cut[-n-2]
                    else:
                        break

                bad_pixels.append( (yvals[-1] < ymin[-1]) | (yvals[-1] > ymax[-1]) | (xvals < xmin) | (xvals > xmax))
                # We fit a polynomial to extract the trace:
                fitted_polymodel.append(linfitter(polymodel, xvals[~bad_pixels[-1]], yvals[-1][~bad_pixels[-1]]))
        else:
            raise RuntimeError("Wrong method name. Accepted entries: 'argmax', 'moments' or 'multi'.")

        if main_plot:
            image_shape = np.shape(target_spec_data)
            aspect_ratio = image_shape[0]/image_shape[1]
            plt.figure(figsize=(12,12*aspect_ratio)) 
            plt.imshow(np.log10(target_spec_data), vmax=np.log10(target_spec_data.max()), cmap = "gray", origin='lower')
            plt.gca().set_aspect(1)
            if len(final_peak_positions) > 1:
                plt.title(f"We have a total of {len(final_peak_positions)} detected spectra")
            else:
                plt.title(f"We have detected {len(final_peak_positions)} spectrum")
            for n in range(len(fitted_polymodel)):
                plt.plot(xvals, fitted_polymodel[n](xvals), 'C0')
                plt.fill_between(xvals, fitted_polymodel[n](xvals)-trace_half_width, fitted_polymodel[n](xvals)+trace_half_width, color='C1', alpha=0.3)
                plt.text(target_spec_data.shape[1]*0.1, final_peak_positions[n], f"Spec {n}")

        if plot_residuals:
            for n in range(len(fitted_polymodel)):
                plt.figure(figsize=(12,12*aspect_ratio)) 
                plt.imshow(np.log10(target_spec_data[ymin[n]:ymax[n],:]), extent=[0,target_spec_data.shape[1],ymin[n],ymax[n]], vmax=np.log10(target_spec_data[ymin[n]:ymax[n],:].max()), cmap = "gray", origin='lower')
                plt.gca().set_aspect(10)
                plt.plot(xvals, fitted_polymodel[n](xvals), 'C0')
                plt.axis((0,len(target_spec_data[0,:]),ymin[n],ymax[n]))
                plt.fill_between(xvals, fitted_polymodel[n](xvals)-trace_half_width, fitted_polymodel[n](xvals)+trace_half_width, color='C1', alpha=0.15)
                plt.title(f"Spectrum {n} - "+self.target_name+" field")

                plt.figure(figsize=(12,12*aspect_ratio))
                plt.plot(xvals[~bad_pixels[n]], yvals[n][~bad_pixels[n]] - fitted_polymodel[n](xvals[~bad_pixels[n]]), 'x')
                plt.plot([xvals.min(),xvals.max()],[0,0],"k--")
                plt.ylabel("Trace residuals (data-model)")
                plt.title(f"Spectrum {n} - Trace residuals")

        if main_plot or plot_residuals:
            plt.show()

        return fitted_polymodel
    
    
    def extracting(self):

        """
        Function that takes the trace as input to extract the spectrum.

        Usar trace_half_width aqui de novo.
        
        """